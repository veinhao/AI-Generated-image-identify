def check_accuracy_and_find_errors(file_path):
    correct_predictions = 0
    total_predictions = 0
    error_lines = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # 移除最后一行
        lines = lines[:-1]

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # 跳过格式不正确的行

            file_name, prediction = parts
            if ("fake" in file_name and prediction == "0") or ("real" in file_name and prediction == "1"):
                correct_predictions += 1
            else:
                error_lines.append(line.strip())
            total_predictions += 1

    if total_predictions == 0:
        return 0, []  # 防止除以零

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, error_lines

# 使用示例
file_path = '/home/nano/PycharmProjects/AI-Generated-image-identify/output/山东中医药大学_王文浩_1126_score.txt'  # 替换为你的文件路径
accuracy, errors = check_accuracy_and_find_errors(file_path)
print(f"正确率: {accuracy:.2f}%")
if errors:
    print("错误的预测行:")
    for error in errors:
        print(error)
