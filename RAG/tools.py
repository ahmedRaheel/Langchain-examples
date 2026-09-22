import numpy as np

def get_file_text(filename) -> str |None:
    try:
        with open(filename, "r") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return None
    
def cosine_similarity(vector1, vector2):

    v1 = np.array(vector1)
    v2 = np.array(vector2)

    return np.dot(v1, v2) / (
        np.linalg.norm(v1) *
        np.linalg.norm(v2)
    )

