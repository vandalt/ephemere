# Keys from the archive
NAME_KEY = "pl_name"
PER_KEY = "pl_orbper"
TP_KEY = "pl_orbtper"
TC_KEY = "pl_tranmid"
ECC_KEY = "pl_orbeccen"
OMEGA_KEY = "pl_orblper"
K_KEY = "pl_rvamp"
TRANSIT_FLAG = "tran_flag"

# Keys used to model orbit
ORB_KEYS = [PER_KEY, TP_KEY, ECC_KEY, OMEGA_KEY, K_KEY]
ORB_KEYS_REFS = [ok + "_reflink" for ok in ORB_KEYS]
ORB_KEYS_ERRS = [ok + ek for ok in ORB_KEYS for ek in ("err1", "err2")]

CONTROV_FLAG = "pl_controv_flag"
