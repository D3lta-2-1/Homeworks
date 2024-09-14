import csv

def analyse(data):
    esperance = sum(data) / len(data)
    variance = sum(map(lambda x: (esperance - x) ** 2, data)) / len(data)
    ecart_type = variance ** 0.5
    return (esperance, variance, ecart_type)

def main():
    with open("value.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        
        data = []
        for row in reader:
            data.append(float(row["Resistance"]))

        (esperance, variance, ecart_type) = analyse(data)
        print("Esperance:", esperance)
        print("Variance:", variance)
        print("Ecart type:", ecart_type)
    return

if __name__ == "__main__":
    main()