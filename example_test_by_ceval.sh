mkdir xeval
cd xeval
git clone https://github.com/yixin-zhu/xeval.git
cd xeval
pip install -r requirements.txt
python code/evaluator_series/eval.py --model_name moss --ntrain 5 -t "xiezhi_inter_chn" 
python code/evaluator_series/eval.py --model_name chatglm --ntrain 5 -t "xiezhi_inter_chn" --cuda_device CUDA_INDEX
python code/evaluator_series/eval.py --model_name minimax --minimax_group_id MINIMAX_GROUP_ID --minimax_key MINIMAX_KEY --ntrain 5 -t "xiezhi_inter_chn"