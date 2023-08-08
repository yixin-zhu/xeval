import os
import argparse
import pandas as pd
import torch
from evaluators.moss import Moss_Evaluator
from evaluators.chatglm import ChatGLM_Evaluator
from evaluators.minimax import MiniMax_Evaluator


import time
choices = ["A", "B", "C", "D"]

def main(args):
    # args are parsed from command line
    if "moss" in args.model_name:
        evaluator=Moss_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name
        )
    elif "chatglm" in args.model_name:
        # cuda_device specifies the GPU to use, only for chatglm
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda")
        evaluator=ChatGLM_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name,
            device=device
        )
    else:
        print("Unknown model name")
        return -1

    # subject is the topic to test
    subject_name=args.subject

    # prepare the directory to save the result logs
    if not os.path.exists(r"logs"):
        os.mkdir(r"logs")
    run_date=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    save_result_dir=os.path.join(r"logs",f"{args.model_name}_{run_date}")
    os.mkdir(save_result_dir)
    print(subject_name)

    # load the test data
    val_file_path=os.path.join('data/val',f'{subject_name}_val.csv')
    val_df=pd.read_csv(val_file_path)

    # evaluate the test data
    # if few_shot is True, then the model is trained on the dev data
    if args.few_shot:
        dev_file_path=os.path.join('data/dev',f'{subject_name}_dev.csv')
        dev_df=pd.read_csv(dev_file_path)
        correct_ratio = evaluator.eval_subject(subject_name, val_df, dev_df, few_shot=args.few_shot,save_result_dir=save_result_dir,cot=args.cot)
    else:
        correct_ratio = evaluator.eval_subject(subject_name, val_df, few_shot=args.few_shot,save_result_dir=save_result_dir)
    print("Acc:",correct_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--cot",action="store_true")
    parser.add_argument("--subject","-s",type=str,default="operating_system")
    parser.add_argument("--cuda_device", type=str)    
    args = parser.parse_args()
    main(args)