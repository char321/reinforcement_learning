class User:
    def __init__(self, p_id, clothes):
        self.p_id = p_id
        self.clothes = clothes
        self.pre = None
        self.count = 0

    def get_response(self, cloth, assign_label):
        # function to simulate response of user
        correct_labels = [cloth['bc_id_1'], cloth['bc_id_2']]
        return assign_label in correct_labels

    def guide_label(self, cloth):
        # function to simulate guide of user
        return cloth['bc_id_1']

    def guide_label_with_img(self, img):
        return img['label'][0]

    def get_pid(self):
        return self.p_id

    def get_emotion_level(self, res):
        if self.pre == None:
            self.pre = res
            self.count += 1 if self.count < 6 else 0
            return self.count
        if res == self.pre:
            self.count += 1 if self.count < 6 else 0
            return self.count
        if res != self.pre:
            self.pre = res
            self.count = 0
            return self.count

