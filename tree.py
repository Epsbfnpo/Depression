import os

def display_tree(directory, indent="", is_last=True):
    marker = "└── " if is_last else "├── "
    print(f"{indent}{marker}{os.path.basename(directory)}/")
    indent += "    " if is_last else "│   "
    try:
        items = sorted(os.listdir(directory))
    except PermissionError:
        print(f"{indent}[Access Denied]")
        return
    files = [i for i in items if os.path.isfile(os.path.join(directory, i))]
    dirs = [i for i in items if os.path.isdir(os.path.join(directory, i))]
    for i, d in enumerate(dirs):
        path = os.path.join(directory, d)
        is_last_dir = (i == len(dirs) - 1) and (len(files) == 0)
        display_tree(path, indent, is_last_dir)
    max_files = 10
    for i, f in enumerate(files):
        if i < max_files:
            is_last_file = (i == len(files) - 1)
            file_marker = "└── " if is_last_file else "├── "
            print(f"{indent}{file_marker}{f}")
        elif i == max_files:
            print(f"{indent}└── ... ({len(files) - max_files} more files not shown)")
            break

if __name__ == "__main__":
    root_path = "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/LMVD_Feature"
    if os.path.exists(root_path):
        display_tree(os.path.abspath(root_path))
    else:
        print("Path does not exist. Please check the root_path setting.")