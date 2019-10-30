import math

# TODO: fix find_horizontal marking large block of text as a group due to vertical text.
# TODO: make find_closest_neighbor that uses corner data instead of center data


def find_group(set, key):
    for group in set:
        if group.__contains__(key):
            return group
    return None


def combine_groups(group1, group2):
    new_group = group1

    # loop through all elements of group2
    for txt in group2:

        # cross search against elements already in new_group
        if not new_group.__contains__(txt):
            new_group.append(txt)


def print_group(group):
    for txt in group:
        print(txt.get_text())


def print_set(set):
    for group in set:
        print()
        print_group(group)

'''
    self.texts              - list of all text elements
    self.average_distance   - average distance between the center of one block of text to its closest neighbor
    self.horizontal         - set of blocks of text that share the same y-space
    self.vertical           - set of blocks of text that share the same x-space
    self.unattached         - group of detached texts (calculated using average distance)
'''


class GroupFinder:

    def __init__(self, texts):
        self.texts = texts
        self.horizontal = []
        self.vertical = []
        self.average_distance = 0
        self.detached = []
        self.update_groups()

    def update_groups(self):
        self.average_distance = self.find_average_distance()
        self.horizontal = self.find_horizontal()
        self.vertical = self.find_vertical()
        self.detached = self.find_detached()

        #print('avg-dst:', self.average_distance)
        #print_group(self.detached)

    def find_average_distance(self):

        total_distance = 0
        num_elements = 0

        for txt1 in self.texts:

            distance_to_closest_neighbor = math.inf
            num_elements += 1

            for txt2 in self.texts:
                if txt1 is txt2:
                    continue

                new_distance = txt1.get_distance(txt2)

                if new_distance < distance_to_closest_neighbor:
                    distance_to_closest_neighbor = new_distance

            total_distance += distance_to_closest_neighbor

        return total_distance/num_elements

    def find_horizontal(self):
        set = []

        for txt1 in self.texts:

            # create list 'group' and append txt1
            group = [txt1]

            for txt2 in self.texts:

                y1 = txt1.get_rect().get_y()
                y2 = txt1.get_rect().get_y() + txt1.get_rect().get_height()

                y3 = txt2.get_rect().get_y()
                y4 = txt2.get_rect().get_y() + txt2.get_rect().get_height()

                #   y1 is within txt2    or  y2 within txt2       or  y3 within txt1       or  y4 within txt1
                if (y1 > y3)and(y1 < y4) or (y2 > y3)and(y2 < y4) or (y3 > y1)and(y3 < y2) or (y4 > y1)and(y4 < y2):
                    group.append(txt2)

            set.append(group)

        set = self.simplify_set(set)
        return set

    def find_vertical(self):
        set = []

        for txt1 in self.texts:

            # create list 'group' and append txt1
            group = [txt1]

            for txt2 in self.texts:
                x1 = txt1.get_rect().get_x()
                x2 = txt1.get_rect().get_x() + txt1.get_rect().get_width()

                x3 = txt2.get_rect().get_x()
                x4 = txt2.get_rect().get_x() + txt2.get_rect().get_width()

                #   x1 is within txt2    or  x2 within txt2       or  x3 within txt1       or  x4 within txt1
                if (x1 > x3)and(x1 < x4) or (x2 > x3)and(x2 < x4) or (x3 > x1)and(x3 < x2) or (x4 > x1)and(x4 < x2):
                    group.append(txt2)

            set.append(group)

        set = self.simplify_set(set)
        return set

    def find_detached(self):
        group = []

        for txt1 in self.texts:

            distance_to_closest_neighbor = math.inf

            for txt2 in self.texts:
                if txt1 is txt2:
                    continue

                new_distance = txt1.get_distance(txt2)

                if new_distance < distance_to_closest_neighbor:
                    distance_to_closest_neighbor = new_distance

            if distance_to_closest_neighbor > 1 * self.average_distance:
                group.append(txt1)

        return group

    def simplify_set(self, set):
        unique_groups = []

        for unique_text in self.texts:
            unique_group = find_group(unique_groups, unique_text)

            if unique_group is None:
                unique_group = []
                unique_groups.append(unique_group)

            # search all groups for contain the unique rect
            for group in set:
                # search specific group for unique rect
                if group.__contains__(unique_text):
                    combine_groups(unique_group, group)

        return unique_groups

    def get_horizontal(self):
        return self.horizontal

    def get_vertical(self):
        return self.vertical

    def get_detached(self):
        return self.unattached
