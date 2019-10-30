
def remove_overlapping_lists(groups):
    groups = sorted(groups, key=len, reverse=True)

    for group1 in groups:
        for group2 in groups:
            if group1 is not group2:
                if all(elem in group1 for elem in group2):
                    groups.remove(group2)

    return groups
