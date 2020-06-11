from .interfaces import client_authorizer

struct Fraction:
    num: uint256
    denom: uint256

# Events:
ClientEnrolled: event({
    _client: indexed(address),
    client: address,
})

DecryptionKeyRequest: event({
    _dp_round: uint256,
    _auditor: indexed(address),
    _client: indexed(address),
    dp_round: uint256,
    auditor: address,
    client: address
})

DecryptionKeyResponse: event({
    _dp_round: uint256,
    _auditor: indexed(address),
    _client: indexed(address),
    dp_round: uint256,
    auditor: address,
    client: address,
    provider: address,
    decryption_key_ipfs_address: bytes[34]
})

DataSubmitted: event({
    _dp_round: indexed(uint256),
    _client: indexed(address),
    dp_round: uint256,
    client: address,
    data_ipfs_address: bytes[34]
})

DataRetrieved: event({
    _auditor: indexed(address),
    _client: indexed(address),
    _dp_round: indexed(uint256),
    auditor: address,
    client: address,
    dp_round: uint256})

AdvanceToScoringStageEvent: event({
    _client: indexed(address),
    _dp_round: indexed(uint256),
    client: address,
    dp_round: uint256
})

EncryptedScoreSubmitted: event({
    _auditor: indexed(address),
    _client: indexed(address),
    _dp_round: indexed(uint256),
    auditor: address,
    client: address,
    dp_round: uint256,
    encrypted_score: bytes32
})

AdvanceToScoreDecryptingStageEvent: event({
    _client: indexed(address),
    _dp_round: indexed(uint256),
    client: address,
    dp_round: uint256
})

DecryptedScoreSubmitted: event({
    _auditor: indexed(address),
    _client: indexed(address),
    _dp_round: indexed(uint256),
    auditor: address,
    client: address,
    dp_round: uint256,
    decrypted_score: uint256
})

ModelScoreTallied: event({
    _dp_round: indexed(uint256),
    _client: indexed(address),
    _median_score: indexed(uint256),
    dp_round: uint256,
    auditor: address,
    client: address,
    median_score: uint256  # the proposed median score for the client, not the auditor's score
})

ModelMedianScoreCommitted: event({
    _dp_round: indexed(uint256),
    _client: indexed(address),
    dp_round: uint256,
    client: address,
    median_score: uint256
})

DatasetScoreTallied: event({
    _dp_round: indexed(uint256),
    _auditor: indexed(address),
    _client: indexed(address),
    dp_round: uint256,
    auditor: address,
    client: address,
    dataset_score: uint256,
    new_score: uint256
})

AdvancedToNextDPRound: event({
    _dp_round: indexed(uint256),
    _client: indexed(address),
    dp_round: uint256,
    client: address
})

ClientBooted: event({
    _dp_round: indexed(uint256),
    _client: indexed(address),
    dp_round: uint256,
    client: address
})

_num_active_clients: uint256
_encrypted_scores: map(bytes32, bool)

_client_to_stage: map(address, uint256)
_client_to_num_self_retrieves: map(address, uint256)
_client_to_num_other_retreived: map(address, uint256)
_client_to_num_decrypted_scores: map(address, uint256)
_client_to_median_dp_round: map(address, uint256)
_client_to_median: map(address, uint256)
_client_to_dataset_score: map(address, uint256)
_client_to_model_score: map(address, uint256)

_auditor_to_client_to_encrypted_score: map(address, map(address, bytes32))
_auditor_to_client_to_decrypted_score_dp_round: map(address, map(address, uint256))
_auditor_to_client_to_decrypted_score: map(address, map(address, uint256))
_auditor_to_client_to_dataset_score_included: map(address, map(address, bool))
_auditor_to_num_dataset_scores_included: map(address, uint256)

_dp_round_to_num_scoring_clients: map(uint256, uint256)
_dp_round_to_auditor_to_client_retrieved: map(uint256, map(address, map(address, bool)))
_dp_round_to_auditor_to_client_to_proposed_median_to_score_counted: map(uint256, map(address, map(address, map(uint256, bool))))
_dp_round_to_client_to_proposed_median_to_num_lt: map(uint256, map(address, map(uint256, uint256)))
_dp_round_to_client_to_proposed_median_to_num_eq: map(uint256, map(address, map(uint256, uint256)))
_dp_round_to_client_to_proposed_median_to_num_gt: map(uint256, map(address, map(uint256, uint256)))
_dp_round_to_max_model_score: map(uint256, uint256)
# _dp_round_to_total_model_score: map(uint256, uint256)
_dp_round_to_max_dataset_score: map(uint256, uint256)
# _dp_round_to_total_dataset_score: map(uint256, uint256)
_dp_round_to_total_refund_balance: map(uint256, uint256)
_dp_round_to_num_refundable_clients: map(uint256, uint256)
_dp_round_to_client_to_collectable_model_score: map(uint256, map(address, uint256))
_dp_round_to_client_to_collectable_dataset_score: map(uint256, map(address, uint256))

# constants
_num_dp_rounds: uint256
_start_block: uint256
_dp_round_training_blockdelta: uint256
_dp_round_data_retrieval_blockdelta: uint256
_dp_round_scoring_blockdelta: uint256
_dp_round_score_decrypting_blockdelta: uint256

_bond_amount: uint256
_bond_reserve_amount: uint256
_min_agreement_threshold: Fraction

_client_authorizer: address
_refund_fraction: Fraction
_submission_blockdelta: uint256

SCORE_DENOM: constant(uint256) = 1048576  # setting equal to 2^20. Using a power of 2 to allow for bitshifting when computing the financial reward
IS_CLIENT_AUTHORIZED_GAS: constant(uint256) = 50000  # client_authorizer.is_client_authorized must consume less than 50000 gas

@private
@constant
def _abs_diff(value_1: uint256, value_2: uint256) -> uint256:
    # returns abs(value_1 - value_2) without integer underflow
    if value_1 > value_2:
        return value_1 - value_2
    else:
        return value_2 - value_1

@public
@constant
def get_num_scoring_clients(dp_round: uint256) -> uint256:
    return self._dp_round_to_num_scoring_clients[dp_round]

@public
@constant
def is_retrieved(dp_round: uint256, auditor: address, client: address) -> bool:
    return self._dp_round_to_auditor_to_client_retrieved[dp_round][auditor][client]

# @public
# @constant
# def is_proposed_median_score_counted(dp_round: uint256, auditor: address, client: address, proposed_median: uint256) -> bool:
#     # TODO there's a bug, such that including this next line, causes the contract deploy gas to run out. Not sure why that is happening.
#     return self._dp_round_to_auditor_to_client_to_proposed_median_to_score_counted[dp_round][auditor][client][proposed_median]

# @public
# @constant
# def get_num_lt_scores(dp_round: uint256, client: address, proposed_median: uint256) -> uint256:
#     return self._dp_round_to_client_to_proposed_median_to_num_lt[dp_round][client][proposed_median]

# @public
# @constant
# def get_num_eq_scores(dp_round: uint256, client: address, proposed_median: uint256) -> uint256:
#     return self._dp_round_to_client_to_proposed_median_to_num_eq[dp_round][client][proposed_median]

# @public
# @constant
# def get_num_gt_scores(dp_round: uint256, client: address, proposed_median: uint256) -> uint256:
#     return self._dp_round_to_client_to_proposed_median_to_num_gt[dp_round][client][proposed_median]

# @public
# @constant
# def get_total_model_score(dp_round: uint256) -> uint256:
#     return self._dp_round_to_total_model_score[dp_round]

# @public
# @constant
# def get_total_dataset_score(dp_round: uint256) -> uint256:
#     return self._dp_round_to_total_dataset_score[dp_round]

@public
@constant
def get_refund_balance(dp_round: uint256) -> uint256:
    return self._dp_round_to_total_refund_balance[dp_round]

# @public
# @constant
# def get_collectable_model_score(dp_round: uint256, client: address) -> uint256:
#     return self._dp_round_to_client_to_collectable_model_score[dp_round][client]

# @public
# @constant
# def get_collectable_dataset_score(dp_round: uint256, client: address) -> uint256:
#     return self._dp_round_to_client_to_collectable_dataset_score[dp_round][client]

# @public
# @constant
# def get_encrypted_score(auditor: address, client: address) -> bytes32:
#     return self._auditor_to_client_to_encrypted_score[auditor][client]

# @public
# @constant
# def get_decrypted_score_dp_round(auditor: address, client: address) -> uint256:
#     return self._auditor_to_client_to_decrypted_score_dp_round[auditor][client]

# @public
# @constant
# def get_decrypted_score(auditor: address, client: address) -> uint256:
#     return self._auditor_to_client_to_decrypted_score[auditor][client]

@public
@constant
def is_dataset_score_included(auditor: address, client: address) -> bool:
    return self._auditor_to_client_to_dataset_score_included[auditor][client]

@public
@constant
def get_num_dataset_scores_included(auditor: address) -> uint256:
    return self._auditor_to_num_dataset_scores_included[auditor]

@public
@constant
def is_encrypted_score(score: bytes32) -> bool:
    return self._encrypted_scores[score]

@public
@constant
def get_num_self_retrieves(client: address) -> uint256:
    return self._client_to_num_self_retrieves[client]

@public
@constant
def get_num_other_retreived(client: address) -> uint256:
    return self._client_to_num_other_retreived[client]

@public
@constant
def get_num_decrypted_scores(client: address) -> uint256:
    return self._client_to_num_decrypted_scores[client]

@public
@constant
def get_median_dp_round(client: address) -> uint256:
    return self._client_to_median_dp_round[client]

@public
@constant
def get_median(client: address) -> uint256:
    return self._client_to_median[client]

@public
@constant
def get_model_score(client: address) -> uint256:
    return self._client_to_model_score[client]

@public
@constant
def get_dataset_score(client: address) -> uint256:
    return self._client_to_dataset_score[client]

@public
@constant
def get_submission_blockdelta() -> uint256:
    return self._submission_blockdelta

@public
@constant
def get_bond_amount() -> uint256:
    return self._bond_amount

@public
@constant
def get_bond_reserve_amount() -> uint256:
    return self._bond_reserve_amount

@public
@constant
def get_min_agreement_threshold() -> Fraction:
    return self._min_agreement_threshold

@public
@constant
def get_score_denom() -> uint256:
    return SCORE_DENOM

@public
@constant
def get_client_authorizer() -> address:
    return self._client_authorizer

@public
@constant
def get_refund_fraction() -> Fraction:
    return self._refund_fraction

@private
@constant
def _has_started(block_num: uint256) -> bool:
    return block_num >= self._start_block

@public
@constant
def get_start_block() -> uint256:
    return self._start_block

@public
@constant
def get_dp_round_training_blockdelta() -> uint256:
    return self._dp_round_training_blockdelta

@public
@constant
def get_dp_round_data_retrieval_blockdelta() -> uint256:
    return self._dp_round_data_retrieval_blockdelta

@public
@constant
def get_dp_round_scoring_blockdelta() -> uint256:
    return self._dp_round_scoring_blockdelta

@public
@constant
def get_dp_round_score_decrypting_blockdelta() -> uint256:
    return self._dp_round_score_decrypting_blockdelta

@public
@constant
def get_num_dp_rounds() -> uint256:
    return self._num_dp_rounds

@public
@constant
def get_num_active_clients() -> uint256:
    return self._num_active_clients

@private
@constant
def _get_dp_round(block_num: uint256) -> uint256:
    if not self._has_started(block_num):
        return 0
    assert block_num >= self._start_block, "block_num is in the past"
    elapsed_blockdelta: uint256 = block_num - self._start_block
    dp_round_total_blockdelta: uint256 = self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta + self._dp_round_score_decrypting_blockdelta
    assert dp_round_total_blockdelta > 0, "invariant violation"
    round_without_offset: uint256 = elapsed_blockdelta / dp_round_total_blockdelta
    round_with_offset: uint256 = round_without_offset + 1
    return round_with_offset

@public
@constant
def get_dp_round(block_num: uint256) -> uint256:
    return self._get_dp_round(block_num)

@private
@constant
def _get_dp_round_start_block(dp_round: uint256) -> uint256:
    if dp_round == 0:
        # round 0 is before self._start_block
        return 0
    assert dp_round > 0, "invariant violation"
    dp_round_total_blockdelta: uint256 = self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta + self._dp_round_score_decrypting_blockdelta
    return (dp_round - 1) * dp_round_total_blockdelta + self._start_block

@public
@constant
def get_dp_round_start_block(dp_round: uint256) -> uint256:
    return self._get_dp_round_start_block(dp_round)

@private
@constant
def _get_client_dp_round(client: address) -> uint256:
    client_stage: uint256 = self._client_to_stage[client]
    if client_stage == 0:
        return 0
    assert client_stage > 0, "invariant violation"
    client_stage_without_offset: uint256 = client_stage - 1
    client_round_without_offset: uint256 = (client_stage_without_offset / 4)
    client_round_with_offset: uint256 = client_round_without_offset + 1
    return client_round_with_offset

@public
@constant
def get_client_dp_round(client: address) -> uint256:
    return self._get_client_dp_round(client)

@private
@constant
def _is_active_client(client: address) -> bool:
    return self._client_to_stage[client] > 0

@public
@constant
def is_active_client(client: address) -> bool:
    return self._is_active_client(client)

@private
@constant
def _is_client_training(client: address) -> bool:
    client_stage: uint256 = self._client_to_stage[client]
    if client_stage == 0:
        return False
    client_stage -= 1
    return client_stage % 4 == 0

@public
@constant
def is_client_training(client: address) -> bool:
    return self._is_client_training(client)

@private
@constant
def _is_in_training_submission_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    if dp_round > self._num_dp_rounds:
        return False
    if dp_round == 0:
        return False
    dp_round_start_block: uint256 = self._get_dp_round_start_block(dp_round)
    if block_num < dp_round_start_block:
        return False
    if dp_round_start_block + self._dp_round_training_blockdelta >= self._submission_blockdelta:
        if block_num < dp_round_start_block + self._dp_round_training_blockdelta - self._submission_blockdelta:
            return False  # too early
    if block_num >= dp_round_start_block + self._dp_round_training_blockdelta:
        return False
    return True

@public
@constant
def is_in_training_submission_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    return self._is_in_training_submission_blockdelta(block_num, dp_round)

@private
@constant
def _is_client_retrieving_data(client: address) -> bool:
    client_stage: uint256 = self._client_to_stage[client]
    if client_stage == 0:
        return False
    client_stage -= 1
    return client_stage % 4 == 1

@public
@constant
def is_client_retrieving_data(client: address) -> bool:
    return self._is_client_retrieving_data(client)

@private
@constant
def _is_in_data_retrieval_submission_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    if dp_round > self._num_dp_rounds:
        return False
    if dp_round == 0:
        return False
    dp_round_start_block: uint256 = self._get_dp_round_start_block(dp_round)
    if block_num < dp_round_start_block + self._dp_round_training_blockdelta:
        return False
    if dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta >= self._submission_blockdelta:
        if block_num < dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta - self._submission_blockdelta:
            return False  # too early
    if block_num >= dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta:
        return False
    
    return True

@public
@constant
def is_in_data_retrieval_submission_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    return self._is_in_data_retrieval_submission_blockdelta(block_num, dp_round)


@private
@constant
def _is_in_data_to_scoring_advancement_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    if dp_round > self._num_dp_rounds:
        return False
    if dp_round == 0:
        return False
    dp_round_start_block: uint256 = self._get_dp_round_start_block(dp_round)
    if block_num < dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta:
        return False
    if block_num >= dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + min(self._submission_blockdelta, self._dp_round_scoring_blockdelta):
        return False  # too late
    return True

@private
@constant
def _is_in_scoring_to_score_decrypting_advancement_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    if dp_round > self._num_dp_rounds:
        return False
    if dp_round == 0:
        return False
    dp_round_start_block: uint256 = self._get_dp_round_start_block(dp_round)
    if block_num < dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta:
        return False
    if block_num >= dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta + min(self._submission_blockdelta, self._dp_round_score_decrypting_blockdelta):
        return False  # too late
    
    return True

@private
@constant
def _is_in_dp_round_advancement_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    if dp_round > self._num_dp_rounds:
        return False
    if dp_round == 0:
        return False
    next_dp_round_start_block: uint256 = self._get_dp_round_start_block(dp_round + 1)
    if block_num < next_dp_round_start_block:
        return False
    if block_num >= next_dp_round_start_block + min(self._submission_blockdelta, self._dp_round_training_blockdelta):
        return False  # too late
    
    return True

@private
@constant
def _is_client_scoring(client: address) -> bool:
    client_stage: uint256 = self._client_to_stage[client]
    if client_stage == 0:
        return False
    client_stage -= 1
    return client_stage % 4 == 2

@public
@constant
def is_client_scoring(client: address) -> bool:
    return self._is_client_scoring(client)

@private
@constant
def _is_in_scoring_submission_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    if dp_round > self._num_dp_rounds:
        return False
    if dp_round == 0:
        return False
    dp_round_start_block: uint256 = self._get_dp_round_start_block(dp_round)
    if block_num < dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta:
        return False
    if dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta >= self._submission_blockdelta:
        if block_num < dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta - self._submission_blockdelta:
            return False  # too early
    if block_num >= dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta:
        return False
    return True

@public
@constant
def is_in_scoring_submission_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    return self._is_in_scoring_submission_blockdelta(block_num, dp_round)

@private
@constant
def _is_client_score_decrypting(client: address) -> bool:
    client_stage: uint256 = self._client_to_stage[client]
    if client_stage == 0:
        return False
    client_stage -= 1
    return client_stage % 4 == 3

@public
@constant
def is_client_score_decrypting(client: address) -> bool:
    return self._is_client_score_decrypting(client)

@private
@constant
def _is_in_score_decrypting_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    if dp_round > self._num_dp_rounds:
        return False
    if dp_round == 0:
        return False
    dp_round_start_block: uint256 = self._get_dp_round_start_block(dp_round)
    if block_num < dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta:
        return False
    if dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta + self._dp_round_score_decrypting_blockdelta >= self._submission_blockdelta:
        if block_num < dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta + self._dp_round_score_decrypting_blockdelta - self._submission_blockdelta:
            return False  # too early
    if block_num >= dp_round_start_block + self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta + self._dp_round_score_decrypting_blockdelta:
        return False
    return True

@public
@constant
def is_in_score_decrypting_blockdelta(block_num: uint256, dp_round: uint256) -> bool:
    return self._is_in_score_decrypting_blockdelta(block_num, dp_round)

@public
def __init__(
    num_dp_rounds: uint256,
    bond_amount: uint256,
    bond_reserve_amount: uint256,
    start_block: uint256,
    submission_blockdelta: uint256,
    dp_round_training_blockdelta: uint256,
    dp_round_data_retrieval_blockdelta: uint256,
    dp_round_scoring_blockdelta: uint256,
    dp_round_score_decrypting_blockdelta: uint256,
    min_agreement_threshold: Fraction,
    refund_fraction: Fraction,
    client_authorizer: address):

    self._num_dp_rounds = num_dp_rounds

    self._bond_amount = bond_amount
    assert bond_reserve_amount <= bond_amount, "reserve not <= bond"
    self._bond_reserve_amount = bond_reserve_amount

    assert start_block >= block.number, "start_block in past"
    self._start_block = start_block

    assert dp_round_training_blockdelta > 0, "training blockdelta in past"
    self._dp_round_training_blockdelta = dp_round_training_blockdelta

    assert dp_round_data_retrieval_blockdelta > 0, "blockdelta for data retrieval must be positive"
    self._dp_round_data_retrieval_blockdelta = dp_round_data_retrieval_blockdelta

    assert dp_round_scoring_blockdelta > 0, "scoring blockdelta in past"
    self._dp_round_scoring_blockdelta = dp_round_scoring_blockdelta

    assert dp_round_score_decrypting_blockdelta > 0, "blockdelta for decrypting must be positive"
    self._dp_round_score_decrypting_blockdelta = dp_round_score_decrypting_blockdelta

    assert min_agreement_threshold.denom != 0, "invalid fraction"
    assert min_agreement_threshold.num <= min_agreement_threshold.denom, "fraction must be on [0,1]"
    self._min_agreement_threshold = min_agreement_threshold

    assert refund_fraction.denom != 0, "invalid fraction"
    assert refund_fraction.num <= refund_fraction.denom, "fraction must be on [0,1]"
    self._refund_fraction = refund_fraction

    assert submission_blockdelta > 0, "submission_blockdelta must be > 0"
    self._submission_blockdelta = submission_blockdelta

    self._client_authorizer = client_authorizer

@public
@payable
def enroll_client():
    assert not self._has_started(block.number), "already started; too late to join"
    assert msg.value == self._bond_amount, "Incorrect bond"
    assert self._client_to_stage[msg.sender] == 0, "already enrolled"
    enrollment_start_block: uint256 = 0
    if self._start_block >= self._submission_blockdelta:
        enrollment_start_block = self._start_block - self._submission_blockdelta
    assert block.number > enrollment_start_block, "too early"
    if self._client_authorizer != ZERO_ADDRESS:
        assert client_authorizer(self._client_authorizer).is_client_authorized(msg.sender, gas=IS_CLIENT_AUTHORIZED_GAS), "client not authorized"
    self._client_to_stage[msg.sender] = 1
    self._num_active_clients += 1
    log.ClientEnrolled(msg.sender, msg.sender)

@public
def submit_data(data_ipfs_address: bytes[34]):
    dp_round: uint256 = self._get_dp_round(block.number)
    assert dp_round <= self._num_dp_rounds, "no more rounds"
    assert dp_round == self._get_client_dp_round(msg.sender), "client at wrong round"
    assert self._is_client_training(msg.sender), "client not in training stage"
    assert self._is_in_training_submission_blockdelta(block.number, dp_round), "not in training stage"
    self._client_to_num_self_retrieves[msg.sender] = 0
    self._client_to_num_other_retreived[msg.sender] = 0
    self._client_to_stage[msg.sender] += 1
    log.DataSubmitted(dp_round, msg.sender, dp_round, msg.sender, data_ipfs_address)

@public
@constant
def request_decryption_key(dp_round: uint256, client: address):
    log.DecryptionKeyRequest(dp_round, msg.sender, client, dp_round, msg.sender, client)

@public
@constant
def provide_decryption_key(dp_round: uint256, requester: address, client: address, key_ipfs_address: bytes[34]):
    log.DecryptionKeyResponse(dp_round, requester, client, dp_round, requester, client, msg.sender, key_ipfs_address)

@public
def mark_data_retrieved(client: address):
    # assert that both of you are in the data retrieval stage
    dp_round: uint256 = self._get_dp_round(block.number)
    assert self._is_in_data_retrieval_submission_blockdelta(block.number, dp_round), "wrong contract stage"
    assert dp_round == self._get_client_dp_round(msg.sender), "auditor at wrong round"
    assert dp_round == self._get_client_dp_round(client), "client at wrong round"
    assert self._is_client_retrieving_data(msg.sender), "auditor not retrieving data"
    assert self._is_client_retrieving_data(client), "client not retrieving data"
    # if a client dropped, make sure to boot em first. Booting is only available during the data retrieval stage

    assert not self._dp_round_to_auditor_to_client_retrieved[dp_round][msg.sender][client], "already marked as retrieved"
    self._dp_round_to_auditor_to_client_retrieved[dp_round][msg.sender][client] = True
    self._client_to_num_self_retrieves[msg.sender] += 1
    self._client_to_num_other_retreived[client] += 1

    log.DataRetrieved(msg.sender, client, dp_round, msg.sender, client, dp_round)

@public
def advance_to_scoring_stage():
    dp_round: uint256 = self._get_dp_round(block.number)
    assert self._is_in_data_to_scoring_advancement_blockdelta(block.number, dp_round), "not currently scoring"
    assert dp_round == self._get_client_dp_round(msg.sender), "auditor at wrong round"
    assert self._is_client_retrieving_data(msg.sender), "auditor not from data stage"
    cutoff: uint256 = (self._num_active_clients * self._min_agreement_threshold.num) / self._min_agreement_threshold.denom
    assert self._client_to_num_self_retrieves[msg.sender] >= cutoff, "self not retrieve enough"
    assert self._client_to_num_other_retreived[msg.sender] >= cutoff, "others not retrieve self"

    self._client_to_num_decrypted_scores[msg.sender] = 0
    self._auditor_to_num_dataset_scores_included[msg.sender] = 0
    self._dp_round_to_num_scoring_clients[dp_round] += 1

    self._client_to_stage[msg.sender] += 1
    log.AdvanceToScoringStageEvent(msg.sender, dp_round, msg.sender, dp_round)

@public
def submit_encrypted_score(client: address, encrypted_score: bytes32):
    # you must submit an encrypted score for all clients
    # assume that you were able to access everyone's dataset that was marked retreived, so you can properly calculate the median
    # only inlclude the datasets for clients who make it to the scoring stage (i.e. emitted an AdvanceToScoringStageEvent)
    # if there's an issue with client, then their scores is 0
    # you must submit something for every client; otherwise you'll fail
    dp_round: uint256 = self._get_dp_round(block.number)
    assert dp_round == self._get_client_dp_round(msg.sender), "auditor at wrong round"
    assert self._is_in_scoring_submission_blockdelta(block.number, dp_round), "not currently scoring"
    assert self._is_client_scoring(msg.sender), "auditor not scoring"
    assert dp_round == self._get_client_dp_round(client), "client at wrong round"
    assert not self._encrypted_scores[encrypted_score], "cannot reuse encrypted score"
    assert self._auditor_to_client_to_encrypted_score[msg.sender][client] == EMPTY_BYTES32, "score already submitted"

    self._auditor_to_client_to_encrypted_score[msg.sender][client] = encrypted_score
    self._encrypted_scores[encrypted_score] = True
    log.EncryptedScoreSubmitted(msg.sender, client, dp_round, msg.sender, client, dp_round, encrypted_score)

@public
def advance_to_score_decrypting_stage():
    dp_round: uint256 = self._get_dp_round(block.number)
    assert self._is_in_scoring_to_score_decrypting_advancement_blockdelta(block.number, dp_round), "not score decrypting adv"
    assert dp_round == self._get_client_dp_round(msg.sender), "auditor at wrong round"
    assert self._is_client_scoring(msg.sender), "auditor not scoring"

    self._client_to_dataset_score[msg.sender] = SCORE_DENOM
    self._client_to_stage[msg.sender] += 1
    log.AdvanceToScoreDecryptingStageEvent(msg.sender, dp_round, msg.sender, dp_round)

@private
@constant
def _encrypt_score(salt: bytes32, score: uint256) -> bytes32:
    assert salt != EMPTY_BYTES32, "salt cannot be empty"
    assert score <= SCORE_DENOM, "score must be on [0, SCORE_DENOM]" 
    score_bytes: bytes32 = convert(score, bytes32)
    data_combined: bytes[64] = concat(salt, score_bytes)
    digest: bytes32 = keccak256(data_combined)
    return digest

@public
@constant
def encrypt_score(salt: bytes32, score: uint256) -> bytes32:
    return self._encrypt_score(salt, score)

@public
def submit_decrypted_score(client: address, salt: bytes32, score: uint256):
    dp_round: uint256 = self._get_dp_round(block.number)
    assert dp_round == self._get_client_dp_round(msg.sender), "auditor at wrong round"
    assert self._is_in_score_decrypting_blockdelta(block.number, dp_round), "not score decrypt blockdelta"
    assert self._is_client_score_decrypting(msg.sender), "auditor not score decrypting"
    assert dp_round == self._get_client_dp_round(client), "client at wrong round"
    assert self._is_client_scoring(client) or self._is_client_score_decrypting(client), "client before scoring"
    encrypted_score: bytes32 = self._auditor_to_client_to_encrypted_score[msg.sender][client]
    assert encrypted_score != EMPTY_BYTES32, "encrypted score is zero"
    digest: bytes32 = self._encrypt_score(salt, score)
    assert encrypted_score == digest, "digest mismatch"

    # record the score
    self._auditor_to_client_to_encrypted_score[msg.sender][client] = EMPTY_BYTES32  # clearing the score so you can't count it twice
    self._auditor_to_client_to_decrypted_score[msg.sender][client] = score
    self._auditor_to_client_to_decrypted_score_dp_round[msg.sender][client] = dp_round
    self._client_to_num_decrypted_scores[client] += 1
    self._encrypted_scores[encrypted_score] = False
    self._auditor_to_client_to_dataset_score_included[msg.sender][client] = False
    log.DecryptedScoreSubmitted(msg.sender, client, dp_round, msg.sender, client, dp_round, score)

@public
def tally_model_score(auditor: address, client: address, proposed_median: uint256):
    # call for every decrypted score
    next_dp_round: uint256 = self._get_dp_round(block.number)
    assert next_dp_round > 0, "calling too soon"
    dp_round: uint256 = next_dp_round - 1
    assert self._is_in_dp_round_advancement_blockdelta(block.number, dp_round), "not post dp round"
    assert dp_round == self._get_client_dp_round(auditor), "auditor at wrong round"
    assert self._get_client_dp_round(client) == dp_round, "client at wrong round"

    assert not self._dp_round_to_auditor_to_client_to_proposed_median_to_score_counted[dp_round][auditor][client][proposed_median], "auditor already counted"
    assert self._auditor_to_client_to_decrypted_score_dp_round[auditor][client] == dp_round, "score at wrong round"
    auditor_score: uint256 = self._auditor_to_client_to_decrypted_score[auditor][client]
    if auditor_score < proposed_median:
        self._dp_round_to_client_to_proposed_median_to_num_lt[dp_round][client][proposed_median] += 1
    elif auditor_score == proposed_median:
        self._dp_round_to_client_to_proposed_median_to_num_eq[dp_round][client][proposed_median] += 1
    else:
        self._dp_round_to_client_to_proposed_median_to_num_gt[dp_round][client][proposed_median] += 1
    self._dp_round_to_auditor_to_client_to_proposed_median_to_score_counted[dp_round][auditor][client][proposed_median] = True
    log.ModelScoreTallied(dp_round, client, proposed_median, dp_round, auditor, client, proposed_median)

@public
def commit_model_median_score(client: address, proposed_median: uint256):
    next_dp_round: uint256 = self._get_dp_round(block.number)
    assert next_dp_round > 0, "calling too soon"
    dp_round: uint256 = next_dp_round - 1
    assert self._is_in_dp_round_advancement_blockdelta(block.number, dp_round), "not post dp round"
    # all median scores must be set before clients advance, hence == is appropriate for the line below
    assert self._get_client_dp_round(client) == dp_round, "client at wrong round"
    assert self._client_to_median_dp_round[client] < dp_round, "median already set"

    num_lt: uint256 = self._dp_round_to_client_to_proposed_median_to_num_lt[dp_round][client][proposed_median]
    num_eq: uint256 = self._dp_round_to_client_to_proposed_median_to_num_eq[dp_round][client][proposed_median]
    num_gt: uint256 = self._dp_round_to_client_to_proposed_median_to_num_gt[dp_round][client][proposed_median]
    assert num_lt + num_eq + num_gt == self._client_to_num_decrypted_scores[client], "didn't count everyone's score"
    assert num_eq + num_gt > num_lt, "eq + gt not > lt"
    assert num_eq + num_lt >= num_gt, "eq + lt not >= gt"
    assert num_eq > 0, "eq == 0"
    self._client_to_median[client] = proposed_median
    self._client_to_model_score[client] = proposed_median
    self._client_to_median_dp_round[client] = dp_round
    self._dp_round_to_max_model_score[dp_round] = max(self._dp_round_to_max_model_score[dp_round], proposed_median)
    log.ModelMedianScoreCommitted(dp_round, client, dp_round, client, proposed_median)

@public
def tally_dataset_score(client: address):
    next_dp_round: uint256 = self._get_dp_round(block.number)
    assert next_dp_round > 0, "calling too soon"
    dp_round: uint256 = next_dp_round - 1
    assert self._is_in_dp_round_advancement_blockdelta(block.number, dp_round), "not post dp round"
    # clients can advance before all dataset scores are tallied; hence >= is appropriate
    assert self._get_client_dp_round(client) >= dp_round, "client at wrong round"
    assert self._client_to_median_dp_round[client] == dp_round, "client median not set"
    assert self._client_to_median_dp_round[msg.sender] == dp_round, "auditor score not set"
    assert self._auditor_to_client_to_decrypted_score_dp_round[msg.sender][client] == dp_round, "auditor score at wrong round"
    assert not self._auditor_to_client_to_dataset_score_included[msg.sender][client], "already tallied client score"
    score_diff: uint256 = self._abs_diff(self._auditor_to_client_to_decrypted_score[msg.sender][client], self._client_to_median[client])
    dataset_score: uint256 = 0
    if score_diff < SCORE_DENOM/2:
        dataset_score = (SCORE_DENOM * (SCORE_DENOM/2 - score_diff))/(SCORE_DENOM/2 + score_diff)
    self._auditor_to_client_to_dataset_score_included[msg.sender][client] = True
    self._auditor_to_num_dataset_scores_included[msg.sender] += 1
    new_client_score: uint256 = min(self._client_to_dataset_score[msg.sender], dataset_score)  # self._client_to_score[msg.sender] is initially set to SCORE_DENOM
    self._client_to_dataset_score[msg.sender] = new_client_score  
    log.DatasetScoreTallied(dp_round, msg.sender, client, dp_round, msg.sender, client, dataset_score, new_client_score)

@public
def advance_to_next_dp_round():
    next_dp_round: uint256 = self._get_dp_round(block.number)
    assert next_dp_round > 0, "calling too soon"
    dp_round: uint256 = next_dp_round - 1
    assert self._is_in_dp_round_advancement_blockdelta(block.number, dp_round), "not post dp round"
    assert dp_round == self._get_client_dp_round(msg.sender), "auditor at wrong round"
    assert self._is_client_score_decrypting(msg.sender), "auditor not score decrypting"
    # if this assertion is true, then you are ready to advance
    assert self._auditor_to_num_dataset_scores_included[msg.sender] == self._dp_round_to_num_scoring_clients[dp_round], "didn't tally all scores"

    model_score: uint256 = self._client_to_model_score[msg.sender]
    dataset_score: uint256 = self._client_to_dataset_score[msg.sender]
    if dataset_score > self._dp_round_to_max_dataset_score[dp_round]:
        self._dp_round_to_max_dataset_score[dp_round] = dataset_score
    self._dp_round_to_total_refund_balance[dp_round] = as_unitless_number(self.balance)
    if dp_round < self._num_dp_rounds:
        # if not at the end, then keep the reserve amount
        self._dp_round_to_total_refund_balance[dp_round] -= self._bond_reserve_amount
    self._dp_round_to_client_to_collectable_model_score[dp_round][msg.sender] = model_score
    self._dp_round_to_client_to_collectable_dataset_score[dp_round][msg.sender] = dataset_score
    self._dp_round_to_num_refundable_clients[dp_round] += 1
    self._client_to_stage[msg.sender] += 1

    log.AdvancedToNextDPRound(dp_round, msg.sender, dp_round, msg.sender)

@public
def boot_client(client: address):
    # clients can be booted if we're in the data_retrieval stage and the client is not at the current dp round
    # If a client gets booted, then they forfeit the remaining part of the bond
    dp_round: uint256 = self._get_dp_round(block.number)
    assert dp_round > 0, "give everyone a chance!"
    assert self._is_in_data_retrieval_submission_blockdelta(block.number, dp_round), "wrong contract stage"
    assert dp_round != self._get_client_dp_round(client), "client at correct round, no need to boot"
    assert self._is_active_client(client), "not a client"

    # BOOTING!
    self._num_active_clients -= 1
    self._client_to_stage[client] = 0
    log.ClientBooted(dp_round, client, dp_round, client)

    # send the bond reserve amount to the booter
    half_bond: uint256 = self._bond_reserve_amount
    send(msg.sender, half_bond)

@public
@nonreentrant("collect_reward")
def collect_reward(dp_round: uint256):
    # issue the refund
    assert block.number >= self._get_dp_round_start_block(dp_round) + self._dp_round_training_blockdelta, "not ahead of training"
    max_model_score: uint256 = self._dp_round_to_max_model_score[dp_round]
    assert max_model_score > 0, "model = 0"
    client_model_score: uint256 = self._dp_round_to_client_to_collectable_model_score[dp_round][msg.sender]
    self._dp_round_to_client_to_collectable_model_score[dp_round][msg.sender] = 0  # clearing the score to prevent double refunds

    max_dataset_score: uint256 = self._dp_round_to_max_dataset_score[dp_round]
    assert max_dataset_score > 0, "dataset = 0"
    client_dataset_score: uint256 = self._dp_round_to_client_to_collectable_dataset_score[dp_round][msg.sender]
    self._dp_round_to_client_to_collectable_dataset_score[dp_round][msg.sender] = 0  # clearing the score to prevent double refunds

    client_model_score_scaled: uint256 = (client_model_score * SCORE_DENOM) / max_model_score
    client_dataset_score_scaled: uint256 = (client_dataset_score * SCORE_DENOM) / max_dataset_score

    client_overall_score: uint256 = min(client_model_score_scaled, client_dataset_score_scaled)

    refund_amount: uint256 = 0
    assert self._dp_round_to_num_refundable_clients[dp_round] > 0, "no refundable clients"
    assert self._refund_fraction.denom > 0, "refund fract denom is 0"
    if dp_round < self._num_dp_rounds:
        refund_amount = (self._refund_fraction.num * self._dp_round_to_total_refund_balance[dp_round] * client_overall_score) / (self._refund_fraction.denom * SCORE_DENOM * self._dp_round_to_num_refundable_clients[dp_round])
    else:
        assert dp_round == self._num_dp_rounds, "invalid dp_round"
        refund_amount = (self._dp_round_to_total_refund_balance[dp_round] * client_overall_score) / (self._dp_round_to_num_refundable_clients[dp_round] * SCORE_DENOM)
    # send(msg.sender, refund_amount)
