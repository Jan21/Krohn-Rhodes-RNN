# callbacks/automaton_extraction.py
from pathlib import Path
import json
import contextlib
import torch
import pytorch_lightning as pl

from .hidden_automaton import (
    minimize_dfa_hopcroft,
    build_dfa_until_fixpoint,
)

def _automaton_to_jsonable(auto):
    """
    Convert an automaton to a JSON-serializable dict.
    Supports either:
      - transitions as {(u, a): v}, or
      - transitions as {u: {a: v}}.
    Ensures keys are strings where required by JSON.
    """
    # states
    if hasattr(auto, "states"):
        states = list(auto.states)
    elif hasattr(auto, "num_states"):
        states = list(range(int(auto.num_states)))
    else:
        # derive from transitions
        S = set()
        trans = getattr(auto, "transitions", {})
        for k, v in trans.items():
            if isinstance(k, tuple) and len(k) == 2:
                S.add(int(k[0]))
            else:
                S.add(int(k))
            if isinstance(v, dict):
                S |= {int(x) for x in v.values()}
            else:
                S.add(int(v))
        states = sorted(S)

    # alphabet
    if hasattr(auto, "alphabet"):
        alphabet = list(auto.alphabet)
    else:
        # derive from labels in tuple-keys form
        alphabet = sorted({k[1] for k in getattr(auto, "transitions", {}) if isinstance(k, tuple)})

    # transitions: normalize to {str(u): {str(a): int(v)}}
    trans_in = getattr(auto, "transitions", {})
    if trans_in and all(isinstance(k, tuple) and len(k) == 2 for k in trans_in):
        trans_out = {}
        for (u, a), v in trans_in.items():
            u_s = str(int(u))
            a_s = str(a)  # labels as strings for JSON dict keys
            v_i = int(v)
            trans_out.setdefault(u_s, {})[a_s] = v_i
    else:
        # assume already nested dict; coerce keys to strings/ints safely
        trans_out = {}
        for u, m in trans_in.items():
            u_s = str(int(u))
            trans_out[u_s] = {str(a): int(v) for a, v in m.items()}

    start_state = int(getattr(auto, "start_state", 0))
    accepting_states = [int(x) for x in getattr(auto, "accepting_states", [])]

    return {
        "states": states,
        "alphabet": [str(a) for a in alphabet],
        "transitions": trans_out,
        "start_state": start_state,
        "accepting_states": accepting_states,
    }

class HiddenAutomatonExtractionCallback(pl.Callback):
    def __init__(
        self,
        *,
        alphabet_symbols: list[str],
        every_n_epochs: int = 5,
        out_dir: str = "prints/snapshots",
        name:str = "Automaton",
        eps: float = 0.2,
        max_len: int = 5,
        cap_per_level: int | None = 512,
        save_checkpoint: bool = True,
        also_every_n_steps: int | None = None,   # optional step-based trigger

    ):
        """
        At the end of each scheduled epoch (and/or step), build a probing finite language,
        extract a hidden-state DFA with build_dfa_from_language, and save it.

        - every_n_epochs: run on epochs k with (k+1) % every_n_epochs == 0
        - also_every_n_steps: optional step trigger, same idea
        - out_dir: where to store JSON/metadata
        - eps: ε-threshold passed to build_dfa_from_language
        - max_len, cap_per_level: language size control
        - save_checkpoint: also persist a trainer checkpoint alongside the extraction
        - alphabet_symbols: override if you don’t want to infer from model/alphabet_size
        """
        self.every_n_epochs = every_n_epochs
        self.also_every_n_steps = also_every_n_steps
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.eps = eps
        self.max_len = max_len
        self.cap_per_level = cap_per_level
        self.save_checkpoint = save_checkpoint
        self.alphabet_symbols = alphabet_symbols

    @torch.no_grad()
    def _run_extraction(self, trainer: pl.Trainer, pl_module: pl.LightningModule, tag: str):
        was_training = pl_module.training
        pl_module.eval()

        # 1) Get core model + vocab
        core_model = pl_module.get_core_for_extraction()

        alphabet_sorted = sorted(list(self.alphabet_symbols))
        symbol_to_idx = {a: i for i, a in enumerate(alphabet_sorted)}
        alphabet_size = len(symbol_to_idx)

        # Run the extraction
        hidden_automaton = build_dfa_until_fixpoint(
            model=core_model,
            symbol_to_idx=symbol_to_idx,
            alphabet_size=alphabet_size,
            eps=self.eps, 
            alphabet_symbols=alphabet_sorted,
            max_states=self.cap_per_level
        )

        hidden_automaton = minimize_dfa_hopcroft(hidden_automaton)


        # 4) Persist results (JSON)
        auto_json = _automaton_to_jsonable(hidden_automaton)
        out_base = self.out_dir+'/'+self.name / f"epoch_step_{tag}"
        out_json = out_base.with_suffix(".json")
        payload = {
            "tag": tag,
            "eps": self.eps,
            "max_len": self.max_len,
            "cap_per_level": self.cap_per_level,
            "alphabet": list(lang.alphabet),
            "automaton":auto_json
        }

        # Ensure JSON-serializable (fallback stringify)
        def _default(o):
            try:
                return dict(o)
            except Exception:
                return str(o)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_default)

        # 6) Optionally save a model checkpoint aligned with this extraction
        if self.save_checkpoint:
            ckpt_path = str(out_base) + ".ckpt"
            # Lightning 2.x: save via the strategy to include stateful things correctly
            trainer.save_checkpoint(ckpt_path)

        # 7) Restore training/eval mode if needed
        if was_training:
            pl_module.train()

        # 8) Optionally log to W&B if present
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            try:
                trainer.logger.experiment.log({"hidden_automaton_snapshot_tag": tag})
                # You could also log the JSON file as an artifact.
            except Exception:
                pass

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch  # 0-based
        if self.every_n_epochs is not None and ((epoch + 1) % self.every_n_epochs == 0):
            self._run_extraction(trainer, pl_module, tag=f"e{epoch+1:04d}")

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        if self.also_every_n_steps is None:
            return
        global_step = trainer.global_step  # increments after optimizer step
        if global_step > 0 and (global_step % self.also_every_n_steps == 0):
            self._run_extraction(trainer, pl_module, tag=f"s{global_step:08d}")