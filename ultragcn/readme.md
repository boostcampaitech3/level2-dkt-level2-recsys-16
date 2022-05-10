# UltraGCN
## Preprocess.py
---
1. Run ultragcn_preprocess.inpyb
2. Run newdata_preprocess.inpyb
3. 생성된 파일들을 data/dkt_withvalid에 저장
    - answerCode.csv
    - item_list.txt
    - neg_no_valid_inference.txt
    - test.txt
    - testilist.csv
    - train_no_valid_inference_data.txt
    - user_list.txt
    - userilist.csv
    - valid.txt
    - valist.csv
* validation을 하지 않을 경우 data/dkt에 저장
    - item_list.txt
    - test.txt
    - neg_train.txt
    - testilist.csv
    - train.txt
    - user_list.txt
    - userilist.csv
* neg_no_valid_inference.txt와 train_no_valid_inference_data.txt 파일면 변경 필요
## ultragcn_withvalid.py
---
1. 파일 설정하기(경로 확인 필요시 경로 확인하기)
    - dkt_config_withvalid.ini
2. 파일 실행
    - python ultragcn_withvalid.py --config_file dkt_config_withvalid.ini
- validation이 없는 경우
    - line 462~480, 485~490 지우기
    - python ultragcn_withvalid.py --config_file dkt_config.ini 
- vaildation 유무를 바꿀 때는 dkt_ii_constraint_mat 파일을 지우고 실행해야함