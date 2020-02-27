class User:
    def __init__(self, p_id, clothes):
        self.p_id = p_id
        self.clothes = clothes

    def get_response(self, cloth, assign_label):
        # function to simulate response of user
        correct_labels = [cloth['bc_id_1'], cloth['bc_id_2']]
        return assign_label in correct_labels

    def guide_label(self, cloth):
        # function to simulate guide of user
        return cloth['bc_id_1']

    def get_pid(self):
        return self.p_id