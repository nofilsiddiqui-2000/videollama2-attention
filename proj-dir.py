import os

# Define root of your project
project_root = os.path.abspath(os.path.dirname(__file__))
print(f"📁 Contents of: {project_root}\n")

# Level 0: Top-level items
for item in os.listdir(project_root):
    item_path = os.path.join(project_root, item)
    print(f"|-- {item}")

    # Level 1: If item is a directory, list its contents
    if os.path.isdir(item_path):
        try:
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                print(f"    |-- {sub_item}")

                # Level 2: If sub_item is a folder, go one more level down
                if os.path.isdir(sub_item_path):
                    try:
                        for sub_sub_item in os.listdir(sub_item_path):
                            print(f"        |-- {sub_sub_item}")
                    except Exception as e:
                        print(f"        [Error reading subfolder: {e}]")

        except Exception as e:
            print(f"    [Error reading folder: {e}]")
