import json


def compare_key_string_equal(file1_path, file2_path, key):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    diff_lines = []
    for line1, line2 in zip(lines1, lines2):
        json1 = json.loads(line1)
        json2 = json.loads(line2)

        if key in json1 and key in json2 and json1[key] != json2[key]:
            diff_lines.append(f"{file1_path}: {json1[key]}")
            diff_lines.append(f"{file2_path}: {json2[key]}")
            diff_lines.append("")

    return diff_lines

def compare_key_sequence_equal(file1_path, file2_path, key):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    def sort_list_of_tuples(lst):
        sorted_lst = sorted(lst, key=lambda x: x[1])
        return sorted_lst

    diff_lines = []
    for line1, line2 in zip(lines1, lines2):
        json1 = json.loads(line1)
        json2 = json.loads(line2)

        list1 = []
        list2 = []
        if key in json1 and key in json2:
            for (k1,v1), (k2,v2) in zip(json1.items(), json2.items()):
                list1.append((k1, v1))
                list2.append((k2, v2))
            sort_list_of_tuples(list1)
            sort_list_of_tuples(list2)

        for list_val1, list_val2 in zip(list1, list2):
            if not (list_val1[0] == list_val2[0]):
                diff_lines.append(f"{file1_path}: {json1[key]}")
                diff_lines.append(f"{file2_path}: {json2[key]}")
                diff_lines.append("")

    return diff_lines

def compare_key_float_equal(file1_path, file2_path, key):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    diff_lines = []
    for line1, line2 in zip(lines1, lines2):
        json1 = json.loads(line1)
        json2 = json.loads(line2)

        if key in json1 and key in json2 and json1[key] != json2[key]:
            diff_lines.append(f"{file1_path}: {json1[key]}")
            diff_lines.append(f"{file2_path}: {json2[key]}")
            diff_lines.append("")

    return diff_lines


def main():
    file1_path = "repeat_test/w1119-90dcba-online-e5845-predict-3.jsonl"
    file2_path = "repeat_test/w1119-90dcba-online-e5845-predict-4.jsonl"
    key1_to_compare = "answer"
    diff_lines = compare_key_string_equal(file1_path, file2_path, key1_to_compare)
    # diff_lines = compare_key_float_equal()

    print(f"count is {len(diff_lines) / 3}")

    if diff_lines:
        print(f"两个文件中 {key1_to_compare} 的值差异:")
        print("\n".join(diff_lines))
    else:
        print("两个文件中没有差异")

if __name__ == "__main__":
    main()



