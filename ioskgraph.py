import requests
import numpy as np
"""def read(uri):
    
    if hasattr(uri, '__iter__'):
        return uri
    else:
        try:
            # try if it is a URL and if we can open it
            f = requests.get(uri).text.split('\n')
        except ValueError:
            # assume it is a file object
            f = open(uri)
        return f
def load_target(name):
    Y = [(y.strip()) for y in read(name) if y]
    print(len(Y))
    Y = [ord(y.strip()) for y in read(name) if y]
    print(Y)
    return np.array(Y)"""

"""import requests

def load_target(name):
    response = requests.get(name)
    response.raise_for_status()  

    Y = []
    for line in response.text.splitlines():
        stripped_line = line.strip()
        if stripped_line:
            for char in stripped_line:
                Y.append(ord(char))
    return np.array(Y)"""

"""def dump_svmlight_file(kernel_matrix, target_array,
                       file_path):
    output = open(file_path, "w")

    for i in range(len(kernel_matrix)):
        output.write(str(target_array[i])+" ")

        for j in range(len(kernel_matrix[i])):
            if kernel_matrix[i][j] != 0.:
                output.write(str(j)+":"+str(kernel_matrix[i][j])+" ")

        output.write("\n")

    output.close()"""
def read(uri):
    """
    Abstract read function. EDeN can accept a URL, a file path and a python list.
    In all cases an iteratatable object should be returned.
    """
    if uri.startswith('http://') or uri.startswith('https://'):
        # Nếu uri là một URL, thực hiện việc lấy nội dung từ trang web
        try:
            response = requests.get(uri)
            response.raise_for_status()  # Nếu có lỗi trong quá trình lấy dữ liệu, sẽ gây ra một exception
            return response.text.split('\n')  # Chia nội dung thành danh sách các dòng và trả về
        except Exception as e:
            print(f"Error retrieving data from URL: {e}")
            return None
    else:
        # Nếu uri không phải là URL, giả định là một đường dẫn đến một tệp tin cục bộ
        try:
            with open(uri, 'r') as file:
                return file.readlines()  # Đọc nội dung từ tệp tin và trả về danh sách các dòng
        except Exception as e:
            print(f"Error reading data from file: {e}")
            return None

def load_target(name):
    Y = [int(y.strip()) for y in read(name) if y]
    print(len(Y))
    print(Y)
    return np.array(Y)

def dump_svmlight_file(kernel_matrix, target_array, file_path):
    output = open(file_path, "w")

    for i in range(len(kernel_matrix)):
        output.write(str(target_array[i])+" ")

        for j in range(len(kernel_matrix[i])):
            if kernel_matrix[i][j] != 0.:
                output.write(str(j)+":"+str(kernel_matrix[i][j])+" ")

        output.write("\n")

    output.close()