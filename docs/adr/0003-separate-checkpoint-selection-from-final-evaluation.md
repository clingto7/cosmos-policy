# Separate checkpoint selection from final evaluation

Select the Cosmos checkpoint and inspect success-rate convergence using fixed pilot simulation seeds. Evaluate the selected checkpoint once on the yk comparison protocol of 50 episodes per task and channel. Reusing those final seeds to choose a checkpoint would bias the reported comparison upward; the pilot and final seed sets must remain separate.
