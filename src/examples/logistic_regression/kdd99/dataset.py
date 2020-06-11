from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array

from ..._utils.clip_scale_transformer import ClipScaleTransformer
from ..dataset import LogisticRegressionDataset

FEATURE_NAMES = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
                 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type']

ATTACK_CLASSIFICATIONS = (
    ("normal.", "normal"),
    ("back.", "dos"),
    ("buffer_overflow.", "u2r"),
    ("ftp_write.", "r2l"),
    ("guess_passwd.", "r2l"),
    ("imap.", "r2l"),
    ("ipsweep.", "probe"),
    ("land.", "dos"),
    ("loadmodule.", "u2r"),
    ("multihop.", "r2l"),
    ("neptune.", "dos"),
    ("nmap.", "probe"),
    ("perl.", "u2r"),
    ("phf.", "r2l"),
    ("pod.", "dos"),
    ("portsweep.", "probe"),
    ("rootkit.", "u2r"),
    ("satan.", "probe"),
    ("smurf.", "dos"),
    ("spy.", "r2l"),
    ("teardrop.", "dos"),
    ("warezclient.", "r2l"),
    ("warezmaster.", "r2l")
)

BINARY_CLASSIFICATION = ((name, name if name == "normal." else "attack.") for name, _ in ATTACK_CLASSIFICATIONS)

KDD99_OUTPUT_CLASSES = np.unique(np.array(("attack.", "normal."), dtype="<U16"))
# OUTPUT_CLASSES = np.unique(np.array(("dos", "normal", "u2r", "r2l", "probe"), dtype="<U16"))
KDD99_NUM_COEFS = 121

class _BinaryYTransformer(TransformerMixin, BaseEstimator):  # type: ignore
    def __init__(self, copy: bool = True) -> None:
        self.copy = copy

    def fit(self, ignored_x: np.ndarray, ignored_y: Optional[np.ndarray] = None) -> '_BinaryYTransformer':
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = check_array(x, copy=self.copy, dtype='<U16', force_all_finite=False)
        x[x[:, 0] != "normal."] = "attack."
        return x

class _MultiYTransformer(TransformerMixin, BaseEstimator):  # type: ignore
    def __init__(self, copy: bool = True) -> None:
        self.copy = copy

    def fit(self, ignored_x: np.ndarray, ignored_y: Optional[np.ndarray] = None) -> '_MultiYTransformer':
        # since the mapping is supplied, nothing to fit
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = check_array(x, copy=self.copy, dtype='<U16', force_all_finite=False)
        a_c = dict(ATTACK_CLASSIFICATIONS)
        unique_vals = np.unique(x)
        for output in unique_vals:
            if output in a_c:
                x[x == output] = a_c[output]
            else:
                x[x == output] = "other_attack"
        return x

_X_S_MAPPER = DataFrameMapper([
    (["duration"], ClipScaleTransformer(0.0, 58329.0, copy=True)),
    (["protocol_type"], OneHotEncoder(categories=[["tcp", "udp", "icmp"]])),
    (["service"], OneHotEncoder(categories=[["http", "smtp", "finger", "domain_u", "auth", "telnet", "ftp", "eco_i", "ntp_u", "ecr_i", "other", "private",
                                             "pop_3", "ftp_data", "rje", "time", "mtp", "link", "remote_job", "gopher", "ssh", "name", "whois", "domain",
                                             "login", "imap4", "daytime", "ctf", "nntp", "shell", "IRC", "nnsp", "http_443", "exec", "printer", "efs",
                                             "courier", "uucp", "klogin", "kshell", "echo", "discard", "systat", "supdup", "iso_tsap", "hostnames",
                                             "csnet_ns", "pop_2", "sunrpc", "uucp_path", "netbios_ns", "netbios_ssn", "netbios_dgm", "sql_net", "vmnet",
                                             "bgp", "Z39_50", "ldap", "netstat", "urh_i", "X11", "urp_i", "pm_dump", "tftp_u", "tim_i", "red_i",
                                             'http_2784', 'aol', 'http_8001', 'harvest']], handle_unknown='ignore')),
    (["flag"], OneHotEncoder(categories=[["SF", "S1", "REJ", "S2", "S0", "S3", "RSTO", "RSTR", "RSTOS0", "OTH", "SH"]])),
    (["src_bytes"], ClipScaleTransformer(0.0, 1.379964e+09, copy=True)),
    (["dst_bytes"], ClipScaleTransformer(0.0, 1.309937e+09, copy=True)),
    (["land"], None),
    (["wrong_fragment"], ClipScaleTransformer(0.0, 3.0, copy=True)),
    (["urgent"], ClipScaleTransformer(0.0, 14.0, copy=True)),
    (["hot"], ClipScaleTransformer(0.0, 77.0, copy=True)),
    (["num_failed_logins"], ClipScaleTransformer(0.0, 5.0, copy=True)),
    (["logged_in"], None),
    (["num_compromised"], ClipScaleTransformer(0.0, 7479.0, copy=True)),
    (["root_shell"], None),
    (["su_attempted"], ClipScaleTransformer(0.0, 2.0, copy=True)),
    (["num_root"], ClipScaleTransformer(0.0, 7468.0, copy=True)),
    (["num_file_creations"], ClipScaleTransformer(0.0, 43.0, copy=True)),
    (["num_shells"], ClipScaleTransformer(0.0, 2.0, copy=True)),
    (["num_access_files"], ClipScaleTransformer(0.0, 9.0, copy=True)),
    (["is_host_login"], None),
    (["is_guest_login"], None),
    (["count"], ClipScaleTransformer(0.0, 511.0, copy=True)),
    (["srv_count"], ClipScaleTransformer(0.0, 511.0, copy=True)),
    (["serror_rate"], None),
    (["srv_serror_rate"], None),
    (["rerror_rate"], None),
    (["srv_rerror_rate"], None),
    (["same_srv_rate"], None),
    (["diff_srv_rate"], None),
    (["srv_diff_host_rate"], None),
    (["dst_host_count"], ClipScaleTransformer(0.0, 255.0, copy=True)),
    (["dst_host_srv_count"], ClipScaleTransformer(0.0, 255.0, copy=True)),
    (["dst_host_same_srv_rate"], None),
    (["dst_host_diff_srv_rate"], None),
    (["dst_host_same_src_port_rate"], None),
    (["dst_host_srv_diff_host_rate"], None),
    (["dst_host_serror_rate"], None),
    (["dst_host_srv_serror_rate"], None),
    (["dst_host_rerror_rate"], None),
    (["dst_host_srv_rerror_rate"], None)
])

_Y_S_MAPPER = DataFrameMapper([
    (["attack_type"], _BinaryYTransformer())
])

def kdd99_csv_to_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, names=FEATURE_NAMES)
    return df

def kdd99_df_to_dataset(df: pd.DataFrame) -> LogisticRegressionDataset:
    xs = _X_S_MAPPER.fit_transform(df)
    ys = _Y_S_MAPPER.fit_transform(df)
    return LogisticRegressionDataset(xs, ys)
