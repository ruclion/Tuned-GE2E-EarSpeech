from synthesizer.hparams import hparams
from synthesizer.FaPig_train import train


if __name__ == "__main__":
    train(run_id='FaPig',
          metadata_fpath=hparams.metadata_fpath,
          models_dir=hparams.models_dir,
          save_every=hparams.save_every,
          backup_every=hparams.backup_every,
          force_restart=hparams.force_restart,
          hparams=hparams,
          )
