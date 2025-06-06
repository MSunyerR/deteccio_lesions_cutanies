"""
Codi utilitzat per filtrar les etiquetes massa petites

"""

import glob


files = glob.glob(".\\dataset\\labels\\*\\*.txt")

print(files)
trobats = 0
for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for i, line in enumerate(lines):

        # yoloBox =[class,x,y,w,h]
        yoloBox = [float(t.strip()) for t in line.split()]
        cx = yoloBox[1]
        cy = yoloBox[2]
        w = yoloBox[3]
        h = yoloBox[4]

        if (w < 0.003 and h < 0.0045) or (w < 0.0015) or (h < 0.002):
            # No afegim la línia, ja que la considerem massa petita
            print("Trobat")
            trobats+=1
            continue
        else:
            new_lines.append(line)

    with open(file, 'w') as f:
        f.writelines(new_lines)

    print(f"Fitxer processat: {file}")
print(trobats)