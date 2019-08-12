



a = ["this", "are", "not"]
b = ["they", "are", "not"]

result = set(a) & set(b)
print(type(list(result)))
print(list(result))




path = "C:/Users/olga-/Desktop/test.csv"

emotion= ":-)"
position = int(2)

class_1 = 0
class_2 = 0
class_3 = 0
class_4 = 0
class_5 = 0

with open(path) as f:
    lines = f.readlines()
    for index, line in enumerate(lines):
        if index >= 1:
            stripped_line = line.split(',')
            print(type(stripped_line[1]))
            if stripped_line[1] == "4":
                if stripped_line[position] != "0.0":
                    class_4 += 1
            if stripped_line[1] == "1":
                if stripped_line[position] != "0.0":
                    class_1 += 1
            if stripped_line[1] == "2":
                if stripped_line[position] != "0.0":
                    class_2 += 1
            if stripped_line[1] == "3":
                if stripped_line[position] != "0.0":
                    class_3 += 1
            if stripped_line[1] == "5":
                if stripped_line[position] != "0.0":
                    class_5 += 1


print(class_1)
print(class_2)
print(class_3)
print(class_4)
print(class_5)




