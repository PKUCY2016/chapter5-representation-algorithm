import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from reference import theorem_sample, group_snr, looped_crate

def layers(U,d):
    return [{"U":U,"D":np.eye(d),"epsilon":.75,"kappa":.01,"eta":.05,"lambda_":.01,"nonnegative":True}]

def main(out):
    rows=[]
    for seed in [0,1,2,10,11,12]:
        X,U,g=theorem_sample(32,2,8,32,.1,.75,seed)
        for R in [1,2,4,8]:
            spec={"R":R,"seed":seed,"task_id":"theorem_toy","layers":layers(U,32)}
            spec["include_state"] = True
            result=looped_crate(X,spec)
            result["input_SNR"]=group_snr(X,U,g)
            result["output_SNR"]=group_snr(np.asarray(result["receipt"]["final_state"]),U,g)
            result["baseline_family"]="loop_tied"
            result["parameter_count"] = 32*32 + 32*16
            result["parameter_match_error"] = 0.0 if R == 1 else None
            result["comparison_status"]="DERIVED_PROTOTYPE"
            rows.append(result)
        # equal-FLOPs untied depth baseline: each layer has the same operator budget.
        spec={"R":1,"seed":seed,"task_id":"equal_flops_untied","layers":layers(U,32)*4,"include_state":True}
        result=looped_crate(X,spec); result["comparison_status"]="DERIVED_PROTOTYPE"; result["input_SNR"]=group_snr(X,U,g); result["output_SNR"]=group_snr(np.asarray(result["receipt"]["final_state"]),U,g); rows.append(result)
        result["baseline_family"]="equal_flops_untied"; result["parameter_count"] = 4*(32*32 + 32*16); result["parameter_match_error"] = None
        # Explicit parameter-matched no-loop baseline: one untied layer.
        spec={"R":1,"seed":seed,"task_id":"parameter_matched_no_loop","layers":layers(U,32),"include_state":True}
        result=looped_crate(X,spec); result["comparison_status"]="DERIVED_PROTOTYPE"; result["baseline_family"]="parameter_matched_no_loop"; result["parameter_count"]=32*32+32*16; result["parameter_match_error"]=0.0; result["input_SNR"]=group_snr(X,U,g); result["output_SNR"]=group_snr(np.asarray(result["receipt"]["final_state"]),U,g); rows.append(result)
    pathlib.Path(out).write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__': main(sys.argv[1])
