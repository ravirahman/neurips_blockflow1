# The Ethereum Blockchain Smart Contract

This `vyper` folder contains the main agreement, [master.vy](master.vy), which has all on-chain logic.

Time in the smart contract is measured in terms of blocks. Blocks, instead of seconds, are used because they are incremental and deterministic.
It also simplifies the logic for event filtering. On the ethereum mainnet, a new block is mined approximately every 15 seconds.

## Building the smart contract
```bash
make clean && make
```

## Initialization and parameters
The `__init__` function takes the following parameters:
* `num_dp_rounds: uint256`: Number of rounds
* `bond_amount: uint256`: Bond (in wei) that each client must post to join
* `bond_reserve_amount: uint256`: A portion of the bond (in wei) that is refunded at the end for successful completion
* `start_block: uint256`: The block at which the experiment starts (enrollment must happen before this deadline)
* `submission_blockdelta: uint256`: This parameter is a bit complicated:
  * To limit the # of blocks one must search for events, clients are allowed to call
    state-changing (and event-emitting) functions no earlier than `submission_blockdelta` before the deadline (for submission actions) or after the waiting
    period (for advancement actions).
  * For example, if a submission action has a deadline at block 100, and `submission_blockdelta` is 50, the client cannot
    invoke the action before block 50. That said, the action may not be available until a later action, because of preconditions
    that are not completed until a later block (but before the deadline).
* `dp_round_training_blockdelta: uint256`: The duration of the training period, in terms of blocks
* `dp_round_data_retrieval_blockdelta: uint256`: The duration of the data retrieval period, in terms of blocks.
* `dp_round_scoring_blockdelta: uint256`: The duration of the score submission period, in terms of blocks.
* `dp_round_score_decrypting_blockdelta: uint256`: The duration of the score decryption period, in terms of blocks
* `min_agreement_threshold: Fraction`: The minimum agreement required for determining whether a dataset could be decrypted and scored properly. Recommended to set to 2/3
* `refund_fraction: Fraction`: The proportion of the remaining funds at the end of each round to refund. See the formula in the contract for how refunds are calculated
* `client_authorizer: address`: The address of a smart contract that implements [interfaces/client_authorizer.vy](interfaces/client_authorizer.vy). Can be set to the `ZERO_ADDRESS`, in
which case all clients are permitted to join (open experiment).

## Flow of contract

It is expected that the clients will interact with the smart contract in the following manner. Note that functions must be called within the limit
mandated by `submission_blockdelta`, as described above:
1. Before the `start_block`, clients must call `enroll_client`, and pass an IPFS cid containing the validation dataset folder. Clients must send `bond_amount` ether with this transaction
2. For each round `i` in `[1, num_dp_rounds]` (yes, rounds are 1-indexed. This is because round 0 is considered to be pre-start):
   1. Train, and call `submit_data(model_ipfs_folder)` before the training deadline for the round.
   2. Retrieve and decrypt others' models and validation datasets. Report whether decryption was successful by calling `mark_data_retrieved(client)` for each active client before the
      data retrieval deadline for the round.
     1. If any client failed to advance to this data retrieval stage (i.e. they are stuck on a previous round), call `boot_client(client)` to kick them out of the experiment. You will get `bond_reserve_amount` for booting them. You can also boot yourself, if you did not follow through during an earlier step, to get back your `bond_reserve_amount`.
   3. Call `advance_to_scoring_stage` immediately after the data retrieval deadline for the round. You will only be allowed to call this method if you marked at least 2/3 of the clients'
      datasets as retrievable, and 2/3 of the clients retrieved your dataset and model successfully. If you cannot advance, you will be eliminated.
   4. Score each client. Call `submit_encrypted_score` for each client before the score encryption deadline for the round. You are required to CORRECTLY score all clients who COULD successfully
      call `advance_to_scoring_stage`, regardless of whether or not they invoked this method.
        1. Convert scores to ints by computing `int(score * 1048576)`
        2. Scores are encrypted via `keccak256(32_byte_salt, 32_byte_bigendian_int_score)`
   5. Call `advance_to_score_decrypting_stage` immediately after the encrypted score submission deadline for the round.
   6. Call `submit_decrypted_score` for each client. You must provide the int score, and the salt, that you used to compute the score in the previous stage.
      1. The smart contract keeps a running tally for the most popular decrypted score.
   7. Immediately after the score decryption deadline, compute the median of the submitted scores for each model. Median is defined as the exact 50th percentile (if there are an odd # of submissions)
      or the score nearest and smaller than the interpolated median.
   8. For each `(auditor, client)` tuple for which `auditor` scored (and decrypted the score for) `client`'s model, call `tally_model_score(auditor, client, median)`. Anyone can call this method.
      Only the first call will succeed, subsequent calls will fail.
   9. After `tally_model_score` has been called for each auditor who decrypted a score for `client`, call `commit_model_median_score(client, median)`. This function will check that the proposed median
      is indeed correct. Anyone can call this method. Only the first call will succeed, subsequent calls will fail.
   10. After calling `commit_model_median_score(client)`, then each auditor needs to call `tally_dataset_score(client)`. Each auditor needs to do this for each client.
   11. Finally, call `advance_to_next_dp_round`. You will not be permitted to call this method if you were not able to call `tally_dataset_score(client)` for each client whose dataset was
       marked as being retrieved successfully. This will advance you to the next round, so you can repeat the process. This must be called before the training deadline of the next round.
       After the training deadline for the next round, you may call  `collect_reward(i-1)` to collect your reward for completing this (now-previous) round.
