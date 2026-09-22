from retiever import search_text


user_input =  input("You:\n")
if not user_input:
    print("no text entered")

file_name, content = search_text(user_input)

print(f"Answer is found in filename {file_name}  and details {content} ")
