#!/bin/bash
# Submit both jobs with dependency:
#   JOB1=$(sbatch --parsable slurm_extract_activations.sh)
#   sbatch --dependency=afterok:$JOB1 slurm_train_sae_cc3m_laion.sh
#
# If extraction is interrupted and you need to re-run it:
#   sbatch slurm_extract_activations.sh        (resumes from existing chunks automatically)
#   sbatch --dependency=afterok:$NEWJOB slurm_train_sae_cc3m_laion.sh
