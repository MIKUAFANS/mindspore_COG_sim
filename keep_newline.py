import shutil
import time


def main():
    shutil.copyfile("my_code/test.txt", "my_code/test_back.txt")
    with open("my_code/test.txt", "r+") as f:
        data = f.readlines()

        # with open("my_code/test_backup.txt", "w") as f_back:
        #     for line in data:
        #         f_back.write(line)

        if "\n" not in data[-1]:
            print("Adding newline")
            f.seek(0, 2)
            f.write("\n")


if __name__ == "__main__":
    while True:
        main()
        time.sleep(1800)