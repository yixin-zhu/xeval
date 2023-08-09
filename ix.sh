mkdir zyx
cd zyx
mkdir xeval
cd xeval
git clone https://github.com/yixin-zhu/xeval.git
cd xeval
pip install -r requirements.txt
python code/evaluator_series/eval.py --model_name moss --ntrain 5 -s "computer_architecture" 
python code/evaluator_series/eval.py --model_name chatglm --few_shot --ntrain 5 -s "computer_architecture" --cuda_device CUDA_INDEX
python code/evaluator_series/eval.py --model_name minimax --minimax_group_id 1690876019571305 --minimax_key eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJOYW1lIjoiQVBJIEtFWTIiLCJTdWJqZWN0SUQiOiIxNjkwODc2MDE5NTg5NzEyIiwiUGhvbmUiOiIiLCJHcm91cElEIjoiMTY5MDg3NjAxOTU3MTMwNSIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6InpodV95aXhpbkBmb3htYWlsLmNvbSIsIkNyZWF0ZVRpbWUiOiIyMDIzLTA4LTAxIDE2OjE4OjEwIiwiaXNzIjoibWluaW1heCJ9.B32WR5YHRh6Snpn88spopJ5vbmC9tA08oOVIB4ZUvrJeTCMNxKgiNa3nzJ5OAdpwrP0E04RPFgebe6g1t5usOYgpDVX0nO6DFbQX7wwRdLSKQ9XT1MIqL4GrJ09bwgNKJ2k40P38rjg_TObVC1aU2UF1f9jmNsrAijVuZ8BAMdRLq35M4hSA-PgeLJDepmqEjzrtYGGcL4Sdz6Sol38098LvdlDkW_jKRwX__d4FWKe30JRJmQ00CoHjwCwZDfnzimiPiRSs4uB5QD9blnjlRwWBQemdsapXjioiDramRjPnM_hwGdGy__t1ENObm-ZXCWDfl2L79zTG6fzKZR_ZWQ --few_shot --ntrain 5 -s "computer_architecture"

