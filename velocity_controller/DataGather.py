import csv
import numpy as np

# Data to be written to the CSV file
data = []

points = 0
print(len(np.arange(-2, 2, 0.05)) ** 3)
for x in np.arange(-2, 2, 0.05):
    for y in np.arange(-2, 2, 0.05):
        for z in np.arange(-2, 2, 0.05):
            if points % 1000000 == 0:
                print(points)

            position = np.array([x, y, z])
            dist = np.linalg.norm(position)
            unit_vector = -position / dist

            vx = unit_vector[0]
            vy = unit_vector[1]
            vz = unit_vector[2]

            vx *= np.clip(dist, -0.25, 0.25)
            vy *= np.clip(dist, -0.25, 0.25)
            vz *= np.clip(dist, -0.25, 0.25)

            data.append([x, y, z, vx, vy, vz])

            points += 1

print(points)

# Create a new CSV file and open it for writing
with open("data.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header row
    csv_writer.writerow(data[0])

    # Write the data rows
    csv_writer.writerows(data[1:])

print("CSV file created successfully!")