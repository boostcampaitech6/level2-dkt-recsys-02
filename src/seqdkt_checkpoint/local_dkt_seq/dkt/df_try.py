data = {
    "test": 0,
    "question": 1,
    "tag": 2,
    "correct": 3,
    # 범주형 변수가 아니므로 +1 X
    # padding을 위해 0이 아닌 다른 값을 사용할 수 있음
    "dffclt": 4,
    "dscrmn": 5,
    "gussng": 6,
    "testTag": 7,
    "user_correct_answer": 7,
    "user_total_answer": 8,
    "user_acc": 9,
    "user_mean": 10,
    'relative_answer_mean': 11,  
    'time_to_solve': 12,
    'time_to_solve_mean': 13,  
    'prior_testTag_frequency': 14
}

past_df={f"past_{k}": v for k, v in data.items()} 
current_df = {f"current_{k}": v for k, v in data.items()}

# for df in [past_df, current_df]:
combined_df = {**past_df, **current_df}
list_key=list(combined_df.keys())

print(list_key)