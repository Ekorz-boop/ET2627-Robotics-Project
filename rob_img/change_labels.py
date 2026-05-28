import os

def update_labels(folder_path, new_label):
    files_modified = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    parts[0] = str(new_label)
                    updated_lines.append(" ".join(parts) + "\n")
                else:
                    updated_lines.append("\n")
            with open(filepath, 'w') as file:
                file.writelines(updated_lines)
            
            files_modified += 1
            print(f"Updated: {filename}")
            
    return files_modified

if __name__ == "__main__":
    folder_directory = '.' 
    target_label = 4 # Change this variable to the integer you want for labels
    
    print(f"Starting label replacement in '{os.path.abspath(folder_directory)}'...")
    total = update_labels(folder_directory, target_label)
    print(f"\nDone! Successfully updated {total} text files to use label '{target_label}'.")