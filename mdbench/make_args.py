n = 1000
with open("args.txt", 'w') as file:
    for verbose in [0]:
        for t in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]:
            for ngpus in [4]:
                for devices in range(1, ngpus+1):
                    args = f"-n {n} -t {t} -ngpus {ngpus} -devices {devices} -verbose {verbose} \n"
                    file.write(args)
