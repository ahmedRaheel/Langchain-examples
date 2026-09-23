from task_planner import  decompose_task

user_request = input("You:\n")

tasks = decompose_task(user_request)

print(tasks)