def normalize_weights(weights):
    s = sum(weights)
    if s == 0:
        return [1.0/len(weights)]*len(weights)
    return [w/s for w in weights]

def weighted_choice(weights):
    cumulative = []
    total = 0
    for w in weights:
        total += w
        cumulative.append(total)
    import random
    r = random.random()*total
    for i, c in enumerate(cumulative):
        if r < c:
            return i
    return len(weights)-1
