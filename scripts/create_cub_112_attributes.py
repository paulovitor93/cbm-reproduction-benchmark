MASK = [
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38,
    40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72,
    75, 80, 84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117,
    119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157,
    158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193,
    194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220,
    221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253,
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294,
    298, 299, 304, 305, 308, 309, 310, 311
]

# Read the original 312 concepts
with open("data/cub/attributes.txt", "r") as f:
    all_attributes = [line.strip().split(" ", 1)[1] for line in f]

# Keep only Koh et al.'s 112 concepts
selected = [all_attributes[i - 1] for i in MASK]

print(len(selected))      # Should print 112
print(selected[:10])

# Save them
with open("data/cub/attributes112.txt", "w") as f:
    for name in selected:
        f.write(name + "\n")

print("attributes112.txt created successfully!")