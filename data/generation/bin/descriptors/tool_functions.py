import numpy as np

def read_tab(tab_file):
    '''
    This function reads the first table genated by the script
    PU_on_PDB.py or by the script of Charlotte PU_in_....
    Returns the first dico with the PUs to keep and their alignments
    '''
    dico = {}
    with open(tab_file,"r") as f:
        for line in f:
            my_line = line.split()
            if len(my_line) != 17:
                continue

            if my_line[1] not in dico.keys():
                dico[my_line[1]] = {}
                dico[my_line[1]]["indexes"] = my_line[8]
                dico[my_line[1]]["residues"] = my_line[3]
                dico[my_line[1]]["residues_idx"] = my_line[4]
                dico[my_line[1]]["structure"] = my_line[-2]
                dico[my_line[1]]["PU"] = my_line[-1]
                dico[my_line[1]]["TM"] = float(my_line[9])
                dico[my_line[1]]["pair_5"] = float(my_line[10])
                dico[my_line[1]]["PB_ID"] = float(my_line[12])
            else:
                if (float(my_line[9]) + float(my_line[10]))/2 > (dico[my_line[1]]["TM"] + dico[my_line[1]]["pair_5"]) /2 :
                    dico[my_line[1]]["indexes"] = my_line[8]
                    dico[my_line[1]]["residues"] = my_line[3]
                    dico[my_line[1]]["residues_idx"] = my_line[4]
                    dico[my_line[1]]["structure"] = my_line[-2]
                    dico[my_line[1]]["PU"] = my_line[-1]
                    dico[my_line[1]]["TM"] = float(my_line[9])
                    dico[my_line[1]]["pair_5"] = float(my_line[10])
                    dico[my_line[1]]["PB_ID"] = float(my_line[12])
                else:
                    continue

    return dico


def create_PUs_alignment(foldrec,PBasigned):
    PUs = foldrec
    ali_dico = {}
    ali_dico_str = {}
    for i in PUs:
        if PUs[i]['PU_pdb'].split('.')[0] not in PBasigned.keys():
            continue
        if PUs[i]['template_PB'].count('Z') == len(PUs[i]['template_PB']):
            try:
                ali_dico[PUs[i]['PU_pdb'].split('.')[0]] = "-" * (PUs[i]['query_start'] - 1) + \
                                                           PBasigned[PUs[i]['PU_pdb'].split('.')[0]][
                                                               'PBassigned'] + "-" * (
                                                                       PUs[i]["size"] - PUs[i]['query_stop'])
                #ali_dico_str[PUs[i]['PU_pdb'].split('.')[0]] = "-" * (PUs[i]['query_start']-1) + PUs[i]['template_AA'] + "-" * (
                #                                                       PUs[i]["size"] - PUs[i]['query_stop'] + PUs[i]['template_AA'].count("-"))
                ali_dico_str[PUs[i]['PU_pdb'].split('.')[0]] = "-" * (PUs[i]['query_start']-1) + PUs[i]['template_AA'] + "-" * (
                                                                       PUs[i]["size"] - PUs[i]['query_stop'])

            except:
                continue
        else:
            ali_dico[PUs[i]['PU_pdb'].split('.')[0]] = "-" * (PUs[i]['query_start'] - 1) + PUs[i][
                'template_PB'] + "-" * (PUs[i]["size"] - PUs[i]['query_stop'])
            #ali_dico_str[PUs[i]['PU_pdb'].split('.')[0]] = "-" * (PUs[i]['query_start']-1) + PUs[i]['template_AA'] + "-" * (
            #        PUs[i]["size"] - PUs[i]['query_stop'] + PUs[i]['template_AA'].count("-"))
            ali_dico_str[PUs[i]['PU_pdb'].split('.')[0]] = "-" * (PUs[i]['query_start']-1) + PUs[i]['template_AA'] + "-" * (
                    PUs[i]["size"] - PUs[i]['query_stop'])

    return (ali_dico , ali_dico_str)


def calculate_SF_representativity(PBasigned):
    dico = {}
    for i in PBasigned:
        for sf in PBasigned[i]['Superfamily']:
            try:
                dico[sf] = dico[sf] + 1
            except:
                dico[sf] = 1
    dico= {k: v / len(PBasigned) for k, v in dico.items()}
    return(dico)





def ranges(nums):
    '''
    Finds the consecutive indexes of numbers
    in order to find the overlapping region between 2
    series of numbers
    '''
    #nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def parse_foldrec(file,ranking):
    '''
    Author: Chris Papadopoulos
    Last modified : 04 / 10 / 2019
    This function parses a foldrec file and keeps the important information
    into a dictionary with different keys.
    You just need to give as input the path of the foldrec file.

    PS: It is possible that you need to modify this function based on your
        project's special needs.
    '''
    # We read the foldrec file
    with open(file, "r") as inp_foldrec:
        dico = {}
        rank = 0
        # Line-by-line
        for nb, line in enumerate(inp_foldrec):
            if nb == 7:
                size = int(line.split()[8].strip())
            # We keep the basic hit infos
            if line.startswith('Alignment :'):
                # Counter of hits
                rank = rank + 1
                if rank not in ranking:
                    continue
                if rank > max(ranking):
                    break
                dico[rank] = {}
                name = line.split('vs')[0].split(':')[1].split(',')[0].strip()
                PU_pdb = line.split('vs')[1].strip()
                # SCOP  = line.split('vs')[1].split(':')[1].strip()
                dico[rank]['name'] = name
                # dico[rank]['SCOP'] = SCOP
                dico[rank]['PU_pdb'] = PU_pdb
                dico[rank]["size"] = size
            # We keep the alignment infos
            if line.startswith('Score :') and rank in ranking:
                # Here we clean the empty strings from the list
                clean_list = filter(None, line.split(' '))
                clean_list_to_use = []
                for elem in clean_list:
                    clean_list_to_use.append(elem)
                # We asign every element to a variable
                Score = clean_list_to_use[2]
                Norm_score = clean_list_to_use[7]
                Query_coverage = clean_list_to_use[12][0:-1]
                Identity = clean_list_to_use[16][0:-1]
                Alignment_length = clean_list_to_use[30]
                dico[rank]['Score'] = float(Score)
                dico[rank]['Norm_score'] = float(Norm_score)
                dico[rank]['Query_coverage'] = float(Query_coverage)
                dico[rank]['Identity'] = float(Identity)
                dico[rank]['Alignment_length'] = int(Alignment_length)

            # We keep the Query infos
            if line.startswith('Query') and rank in ranking:
                clean_list = filter(None, line.split(' '))
                clean_list_to_use = []
                for elem in clean_list:
                    clean_list_to_use.append(elem)

                query_start = clean_list_to_use[1].rstrip()
                query_stop = clean_list_to_use[3].rstrip()
                dico[rank]['query_start'] = int(query_start)
                dico[rank]['query_stop'] = int(query_stop)
                query_seq = clean_list_to_use[2]

                # I try to fund some aminoacids which are not C,E,H,G
                # If there is problem add some more to ensure the output
                if 'W' in sorted(set(list(query_seq))) or \
                        'P' in sorted(set(list(query_seq))) or \
                        'N' in sorted(set(list(query_seq))) or \
                        'R' in sorted(set(list(query_seq))) or \
                        'D' in sorted(set(list(query_seq))) or \
                        'V' in sorted(set(list(query_seq))) or \
                        'L' in sorted(set(list(query_seq))) or \
                        'K' in sorted(set(list(query_seq))) or \
                        'S' in sorted(set(list(query_seq))) or \
                        'T' in sorted(set(list(query_seq))) or \
                        'M' in sorted(set(list(query_seq))) or \
                        'Q' in sorted(set(list(query_seq))) or \
                        'Y' in sorted(set(list(query_seq))):
                    query_AA = query_seq
                    dico[rank]['query_AA'] = query_AA


                elif '9' in sorted(set(list(query_seq))) or \
                        '8' in sorted(set(list(query_seq))) or \
                        '7' in sorted(set(list(query_seq))) or \
                        '6' in sorted(set(list(query_seq))) or \
                        '5' in sorted(set(list(query_seq))) or \
                        '4' in sorted(set(list(query_seq))) or \
                        '3' in sorted(set(list(query_seq))) or \
                        '2' in sorted(set(list(query_seq))) or \
                        '1' in sorted(set(list(query_seq))):
                    confidence = query_seq
                    dico[rank]['query_confidence'] = confidence
                else:
                    psipred_seq = query_seq
                    dico[rank]['query_PSIPRED'] = psipred_seq

            # We keep the template infos
            if line.startswith('Template') and rank in ranking:
                clean_list = filter(None, line.split(' '))
                clean_list_to_use = []
                for elem in clean_list:
                    clean_list_to_use.append(elem)

                template_start = clean_list_to_use[1].rstrip()
                template_stop = clean_list_to_use[3].rstrip()
                dico[rank]['template_start'] = int(template_start)
                dico[rank]['template_stop'] = int(template_stop)
                template_seq = clean_list_to_use[2]

                if 'W' in sorted(set(list(template_seq))) or \
                        'P' in sorted(set(list(template_seq))) or \
                        'N' in sorted(set(list(template_seq))) or \
                        'R' in sorted(set(list(template_seq))) or \
                        'D' in sorted(set(list(template_seq))) or \
                        'V' in sorted(set(list(template_seq))) or \
                        'L' in sorted(set(list(template_seq))) or \
                        'K' in sorted(set(list(template_seq))) or \
                        'S' in sorted(set(list(template_seq))) or \
                        'T' in sorted(set(list(template_seq))) or \
                        'M' in sorted(set(list(template_seq))) or \
                        'Q' in sorted(set(list(template_seq))) or \
                        'Y' in sorted(set(list(template_seq))):
                    template_AA = template_seq
                    dico[rank]['template_AA'] = template_AA


                elif '9' in sorted(set(list(template_seq))) or \
                        '8' in sorted(set(list(template_seq))) or \
                        '7' in sorted(set(list(template_seq))) or \
                        '6' in sorted(set(list(template_seq))) or \
                        '6' in sorted(set(list(template_seq))) or \
                        '5' in sorted(set(list(template_seq))) or \
                        '4' in sorted(set(list(template_seq))) or \
                        '3' in sorted(set(list(template_seq))) or \
                        '2' in sorted(set(list(template_seq))) or \
                        '1' in sorted(set(list(template_seq))):

                    confidence = template_seq
                    dico[rank]['template_confidence'] = confidence
                else:
                    psipred_seq = template_seq
                    dico[rank]['template_PB'] = psipred_seq

    return (dico)


def find_longest_region(structure):
    st = ""
    count = 0
    for x,pos in enumerate(structure):
        if pos == "-":
            st = st + "-"
        elif pos != "-":
            count += 1
            st = st + "|" + str(count) + "|"

    st_regions = list(filter(None, st.split("-")))
    st_regions_new = []
    for i in st_regions:
        st_regions_new.append([int(x) for x in list(filter(None, i.split("|")))])
    longest = st_regions_new[st_regions_new.index(max(st_regions_new, key=len))]
    return longest



def PUs_profile(PUs_alignment):
    profile_size = len(PUs_alignment)
    seq_length = len(PUs_alignment[list(PUs_alignment.keys())[0]])

    mat = {}
    seq = []
    # we loop through every position on the sequence
    for x, pos in enumerate(range(seq_length)):
        mat[x + 1] = []
        dico = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0, "g": 0, "h": 0,
                "i": 0, "j": 0, "k": 0, "l": 0, "m": 0, "n": 0, "o": 0, "p": 0, }
        # we loop through all the PU solutions for this position
        for i in PUs_alignment:
            try:
                dico[PUs_alignment[i][pos]] += 1
            except:
                continue

        if sum(list(dico.values())) / profile_size < 0.33:
            dico = {"a": 0.0625, "b": 0.0625, "c": 0.0625, "d": 0.0625, "e": 0.0625, "f": 0.0625, "g": 0.0625,
                    "h": 0.0625,
                    "i": 0.0625, "j": 0.0625, "k": 0.0625, "l": 0.0625, "m": 0.0625, "n": 0.0625, "o": 0.0625,
                    "p": 0.0625, }
            seq.append("z")

        else:
            position_size = sum(list(dico.values()))
            for z in dico:
                dico[z] = round(dico[z] / position_size, 4)

            max_index = list(dico.values()).index(max(list(dico.values())))
            seq.append(list(dico.keys())[max_index])
        for k in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]:
            mat[x + 1].append(dico[k])

    return (mat, "".join(seq))

def PUs_profile_3states(PUs_alignment):
    profile_size = len(PUs_alignment)
    seq_length = len(PUs_alignment[list(PUs_alignment.keys())[0]])

    mat = {}
    seq = []
    # we loop through every position on the sequence
    for x, pos in enumerate(range(seq_length)):
        mat[x + 1] = []
        dico = {"H":0,"E":0,"C":0,}
        # we loop through all the PU solutions for this position
        for i in PUs_alignment:
            try:
                dico[PUs_alignment[i][pos]] += 1
            except:
                continue

        if sum(list(dico.values())) / profile_size < 0.33:
            dico = {"H":0.3333,"E":0.3333,"C":0.3333,}
            seq.append("z")

        else:
            position_size = sum(list(dico.values()))
            for z in dico:
                dico[z] = round(dico[z] / position_size, 4)

            max_index = list(dico.values()).index(max(list(dico.values())))
            seq.append(list(dico.keys())[max_index])
        for k in ["H","E","C"]:
            mat[x + 1].append(dico[k])

    return (mat, "".join(seq))

def PUs_profile_HCA(PUs_alignment):
    profile_size = len(PUs_alignment)
    seq_length = len(PUs_alignment[list(PUs_alignment.keys())[0]])

    mat = {}
    seq = []
    # we loop through every position on the sequence
    for x, pos in enumerate(range(seq_length)):
        mat[x + 1] = []
        dico = {"1":0,"0":0,".":0,}
        # we loop through all the PU solutions for this position
        for i in PUs_alignment:
            try:
                dico[PUs_alignment[i][pos]] += 1
            except:
                continue

        if sum(list(dico.values())) / profile_size < 0.33:
            dico = {"1":0.3333,"0":0.3333,".":0.3333,}
            seq.append("z")

        else:
            position_size = sum(list(dico.values()))
            for z in dico:
                dico[z] = round(dico[z] / position_size, 4)

            max_index = list(dico.values()).index(max(list(dico.values())))
            seq.append(list(dico.keys())[max_index])
        for k in ["1","0","."]:
            mat[x + 1].append(dico[k])

    return (mat, "".join(seq))

def PUs_profile_HCA_SS(PUs_alignment):
    profile_size = len(PUs_alignment)
    seq_length = len(PUs_alignment[list(PUs_alignment.keys())[0]])

    mat = {}
    seq = []
    # we loop through every position on the sequence
    for x, pos in enumerate(range(seq_length)):
        mat[x + 1] = []
        #dico = {"H":0,"h":0,"E":0,"e":0,"x":0,"M":0,"m":0,}
        dico = {".": 0,"x": 0, }
        # we loop through all the PU solutions for this position
        for i in PUs_alignment:
            try:
                dico[PUs_alignment[i][pos]] += 1
            except:
                continue

        if sum(list(dico.values())) / profile_size < 0.33:
            dico = {".": 0.5,"x": 0.5,}
            seq.append("z")

        else:
            position_size = sum(list(dico.values()))
            for z in dico:
                dico[z] = round(dico[z] / position_size, 4)

            max_index = list(dico.values()).index(max(list(dico.values())))
            seq.append(list(dico.keys())[max_index])
        for k in [".","x"]:
            mat[x + 1].append(dico[k])

    return (mat, "".join(seq))

def read_multiFASTA(fasta_file):
    dico = {}
    with open(fasta_file,'r') as fasta:
        for line in fasta:
            if line.startswith('>'):
                name = str(line.split()[0])[1:]
                dico[name] = ''
            elif line == '\n':
                continue
            else:
                seq = line.strip().replace("*","")
                dico[name] = dico[name] + seq
    return(dico)




NOMENCLATURE_CLASS = {"M": "helix", "MM": "helix", "MMM": "helix", "MMMM": "helix", "MMMMM": "helix", "D": "Beta",
                      "DD": "Beta-sheet", "DDD": "Beta-sheet", "DDDD": "Beta-sheet", "DDDDD": "Beta-sheet",
                      "DMD": "Beta-Alpha-Beta", "DDMD": "Beta-Alpha-Beta", "DMDD": "Beta-Alpha-Beta",
                      "DMMD": "Beta-Alpha-Beta", "DDMMD": "Beta-Alpha-Beta", "DMMDD": "Beta-Alpha-Beta",
                      "DDDMD": "Beta-Alpha-Beta", "DMDDD": "Beta-Alpha-Beta", "DDMDD": "Beta-Alpha-Beta",
                      "DMMMD": "Beta-Alpha-Beta", "DMMMMD": "Beta-Alpha-Beta", "DDMMMDD": "Beta-Alpha-Beta",
                      "DMMMDD": "Beta-Alpha-Beta", "DDMMMD": "Beta-Alpha-Beta", "DMMMMDD": "Beta-Alpha-Beta",
                      "DDDDMD": "Beta-Alpha-Beta", "DDMDDD": "Beta-Alpha-Beta", "DDMMDD": "Beta-Alpha-Beta",
                      "DMDMD": "Beta-Alpha-Beta-Alpha-Beta", "DDMDMD": "Beta-Alpha-Beta-Alpha-Beta",
                      "DMDDMD": "Beta-Alpha-Beta-Alpha-Beta",
                      "DMDMDD": "Beta-Alpha-Beta-Alpha-Beta", "DDMDMDD": "Beta-Alpha-Beta-Alpha-Beta",
                      "DMMDDMD": "Beta-Alpha-Beta-Alpha-Beta", "DMMDMMD": "Beta-Alpha-Beta-Alpha-Beta",
                      "DDMMDMMD": "Beta-Alpha-Beta-Alpha-Beta", "DMMDDMMD": "Beta-Alpha-Beta-Alpha-Beta",
                      "MDM": "Alpha-Beta-Alpha", "MDDM": "Alpha-Beta-Alpha", "MMDM": "Alpha-Beta-Alpha",
                      "MDMM": "Alpha-Beta-Alpha", "MMDDM": "Alpha-Beta-Alpha", "MMDDMM": "Alpha-Beta-Alpha",
                      "MMMDM": "Alpha-Beta-Alpha", "MDMMM": "Alpha-Beta-Alpha", "MDDDM": "Alpha-Beta-Alpha",
                      "MMDDDM": "Alpha-Beta-Alpha", "MMMDMM": "Alpha-Beta-Alpha", "MDDDMM": "Alpha-Beta-Alpha",
                      "MDMD": "Alpha-Beta-Alpha-Beta", "MMDMD": "Alpha-Beta-Alpha-Beta",
                      "MDMMD": "Alpha-Beta-Alpha-Beta", "MMDMMD": "Alpha-Beta-Alpha-Beta",
                      "MMDDMD": "Alpha-Beta-Alpha-Beta", "MMDDMMD": "Alpha-Beta-Alpha-Beta",
                      "MDMMDD": "Alpha-Beta-Alpha-Beta", "MMDMDD": "Alpha-Beta-Alpha-Beta",
                      "MMDDMD": "Alpha-Beta-Alpha-Beta", "DMDM": "Beta-Alpha-Beta-Alpha",
                      "DDMDM": "Beta-Alpha-Beta-Alpha", "DMDDM": "Beta-Alpha-Beta-Alpha",
                      "DMMDM": "Beta-Alpha-Beta-Alpha", "DDMMDM": "Beta-Alpha-Beta-Alpha",
                      "DMMDDM": "Beta-Alpha-Beta-Alpha", "DDMDDM": "Beta-Alpha-Beta-Alpha",
                      "DDMDMM": "Beta-Alpha-Beta-Alpha","DM": "Beta-Alpha", "DDM": "Beta-Alpha", "DDDM": "Beta-Alpha",
                      "DDDDM": "Beta-Alpha","DDDDDM": "Beta-Alpha", "DMM": "Beta-Alpha", "DMMM": "Beta-Alpha",
                      "DMMMM": "Beta-Alpha","DMMMMM": "Beta-Alpha",
                      "DDMM": "Beta-Alpha", "DDDMM": "Beta-Alpha", "DDDDMM": "Beta-Alpha", "DDDMMM": "Beta-Alpha",
                      "DDDDMMMM": "Beta-Alpha", "DDDDMMM": "Beta-Alpha", "DDMMMM": "Beta-Alpha", "MD": "Alpha-Beta",
                      "MMD": "Alpha-Beta", "MMMD": "Alpha-Beta", "MMMMD": "Alpha-Beta", "MMMMMD": "Alpha-Beta",
                      "MDD": "Alpha-Beta", "MDDD": "Alpha-Beta", "MDDDD": "Alpha-Beta", "MDDDDD": "Alpha-Beta",
                      "MMDD": "Alpha-Beta", "MMDDD": "Alpha-Beta",
                      "MMMDDD": "Alpha-Beta", "MMDDDDD": "Alpha-Beta", "MMMMDD": "Alpha-Beta", "MMDDDDD": "Alpha-Beta",
                      "MDMMDM": "Alpha-Beta-Alpha-Beta-Alpha", "MDMDM": "Alpha-Beta-Alpha-Beta-Alpha",
                      "MMDMM": "Aplha-Beta", "DMDMM": "Beta-Alpha-Beta-Alpha","MMDDDD": "Alpha-Beta",
                      "MDMDD": "Alpha-Beta-Alpha-Beta", "MDDMM": "Alpha-Beta-Alpha", "": "Coil",
                      "DMMDMD": "Beta-Alpha-Beta-Alpha-Beta", "DMMDMM": "Beta-Alpha-Beta-Alpha",
                      "MDDMD": "Alpha-Beta-Alpha-Beta", "DMDDDD": "Beta-Alpha-Beta", "DMDDMM": "Beta-Alpha-Beta-Alpha",
                      "MMMDD": "Alpha-Beta", "DDDDDD": "Beta", "DDMMM": "Beta-Alpha",
                      "DDDMDM": "Beta-Alpha-Beta-Alpha", "DMDDDM": "Beta-Alpha-Beta-Alpha",
                      "MDDMMD": "Alpha-Beta-Alpha-Beta"}

# ==================================================================================================================== #

def nomemclature(PU_PB_aligned):
    PU_PB_seq = PU_PB_aligned.replace("-", "")
    PU_PB_converted = ""
    seq = ""
    for i in range(len(PU_PB_seq)):
        if PU_PB_seq[i - 2:i] in ["mm", "lm", "ml", "mn", "nm", "mo", "om", "km", "mk"] \
                or PU_PB_seq[i:i + 2] in ["mm","lm","ml","mn","nm","mo","om","km","mk"]:
            PU_PB_converted += "m"
            seq += "H"
        elif PU_PB_seq[i - 2:i] in ["dd", "ad", "da", "bd", "db", "cd", "dc", "ed", "de", "fd", "df"] \
                or PU_PB_seq[i:i + 2] in ["dd", "ad", "da", "bd", "db", "cd", "dc", "ed", "de", "fd", "df"]:
            PU_PB_converted += "d"
            seq += "E"
        elif PU_PB_seq[i] == "-":
            PU_PB_converted += "-"
            seq += "-"
        else:
            PU_PB_converted += PU_PB_seq[i]
            seq += "C"

    PU_PB_converted_aligned = ""
    seq_aligned = ""
    count = -1
    for x,pos in enumerate(PU_PB_aligned):
        if pos == "-":
            PU_PB_converted_aligned += "-"
            seq_aligned += "-"
        elif pos != "-":
            count += 1
            PU_PB_converted_aligned += PU_PB_converted[count]
            seq_aligned += seq[count]

    cp = 0
    nomemclature_PU = ""
    striped_PU_PB_converted = PU_PB_converted.strip("-").strip("Z").strip("z")
    old = striped_PU_PB_converted[0]

    for i in range(len(striped_PU_PB_converted)):
        if striped_PU_PB_converted[i] == old:
            cp += 1
            if (i + 1) == len(striped_PU_PB_converted):
                if cp >= 2:
                    nomemclature_PU += old.upper()
        if striped_PU_PB_converted[i] != old:
            if cp >= 2:
                nomemclature_PU += old.upper()
            old = striped_PU_PB_converted[i]
            cp = 0
    nomemclature_PU_light = nomemclature_PU.replace("-", "")
    nomemclature_PU_class = "NA"
    if nomemclature_PU_light in NOMENCLATURE_CLASS:
        nomemclature_PU_class = NOMENCLATURE_CLASS[nomemclature_PU_light]

    #return nomemclature_PU, nomemclature_PU_class, seq
    return nomemclature_PU, nomemclature_PU_class, PU_PB_converted_aligned , seq_aligned


def calculate_entopy(profile):
    import math
    import numpy as np
    dico = {}
    entropy = {}
    for pos in profile:
        dico[pos] = []
        for pb in profile[pos]:
            if pb != 0.0:
                dico[pos].append(pb * math.log2(1 / pb))
            else:
                dico[pos].append(0)
        entropy[pos] = round(sum(dico[pos]), 3)

    # Calculate the mean entropy:
    en = []
    for i in entropy:
        if entropy[i] != -np.log2(1/len(profile[1])):
            en.append(entropy[i])
    mean_entropy = np.mean(en)
    # Calculate the RMSD from the mean entropy
    rmsd_list = []
    for j in en:
        rmsd_list.append((mean_entropy - j) ** 2)
    rmsd = np.sqrt(sum(rmsd_list) / len(rmsd_list))

    return (entropy, round(mean_entropy, 4), round(rmsd, 4))



def hydrophobic_perc(sequence):
    hydro = sum(map(sequence.count, ["V","I","L","F","M","Y","W"]))
    return(round(hydro/len(sequence),4))


def count_Topo_SF_Arch(foldrec, PBasigned):
    topologies = {}
    PU_rarity = []
    superfams = {}
    architectures = {}
    PUs = 0
    for rank in foldrec:
        PU = foldrec[rank]['PU_pdb'].replace(".pdb", "")
        try:
            PU_size = len(PBasigned[PU]['PBassigned'])
            PUs += 1
            for i in PBasigned[PU]['Superfamily']:
                try:
                    superfams[i] = superfams[i] + 1
                except:
                    superfams[i] = 1
            for i in PBasigned[PU]['Architecture']:
                try:
                    architectures[i] = architectures[i] + 1
                except:
                    architectures[i] = 1
            PU_rarity.append(1 / len(PBasigned[PU]['Superfamily']))
            PU_portion = (foldrec[rank]['template_stop'] + 1 - foldrec[rank]['template_PB'].count("-") - foldrec[rank][
                'template_start']) / PU_size
            # If the portion of the PU is more than 90% we keep all the topology and continue to the next PU
            if PU_portion >= 0.9:
                try:
                    topologies[PBasigned[PU]['Topology']] = topologies[PBasigned[PU]['Topology']] + 1
                except:
                    topologies[PBasigned[PU]['Topology']] = 1
                continue
            # ----------------------------------------------------- #
            PU_portion_start = (foldrec[rank]['template_start'] - 1) / PU_size
            PU_portion_stop = (foldrec[rank]['template_stop'] - 1) / PU_size
            topo_step = 1 / len(PBasigned[PU]['Topology'])
            topo_step_TMP = topo_step
            topo_str_TMP = ""
            for x, topo in enumerate(PBasigned[PU]['Topology']):
                if topo_step_TMP >= PU_portion_start and topo_step_TMP <= PU_portion_stop:
                    topo_str_TMP = topo_str_TMP + topo
                topo_step_TMP = topo_step_TMP + topo_step
            try:
                topologies[topo_str_TMP] = topologies[topo_str_TMP] + 1
            except:
                topologies[topo_str_TMP] = 1
        except:
            continue

    return topologies, superfams, architectures, PUs


def categiries_entropy(categories: object) -> object:
    import math
    ents = 0
    for i in categories:
        freq = categories[i] / sum(categories.values())
        ent = freq * math.log2((1 / freq))
        ents = ents + ent
    return round(ents / len(categories), 4)


def calculate_proportion_of_seq_disordered(iupred_score):
    count_seg_tmp = 0
    count_agg_seg = 0
    for i, pos in enumerate(iupred_score):
        if pos > 0.5:
            count_seg_tmp += 1
        elif pos <= 0.5 and count_seg_tmp >= 5:
            count_agg_seg = count_agg_seg + count_seg_tmp
            count_seg_tmp = 0
        elif pos <= 0.5 and count_seg_tmp < 5:
            count_seg_tmp = 0
            continue

        if i == len(iupred_score) - 1:
            if count_seg_tmp >= 5:
                count_agg_seg = count_agg_seg + count_seg_tmp
            else:
                continue

    return (round(count_agg_seg / len(iupred_score), 3))


def get_hca_barcode(hca, orf):
    '''
    This module generates the HCA barcode of the total sequence
    Clusters of <= 4 residues are neglected
    '''
    barcode = "." * len(hca.get_seqbin(orf))
    barcode_SS = "." * len(hca.get_seqbin(orf))
    clusters = hca.get_clusters(orf)
    for x, cluster in enumerate(clusters):
        cluster_elements = str(cluster).split('\t')
        if len(cluster_elements[-1]) > 0:
            barcode = barcode[:int(cluster_elements[1]) - 1] + cluster_elements[-1] + barcode[int(cluster_elements[2]):]
        if len(cluster_elements[-1]) > 4:
            # try:
            #    SS_ass     = tool_functions.HCA_clusters_dico[cluster_elements[-1]] * len(cluster_elements[-1])
            #    barcode_SS = barcode_SS[:int(cluster_elements[1]) - 1] + SS_ass + barcode[int(cluster_elements[2]):]
            # except:
            #    SS_ass     = "x" * len(cluster_elements[-1])
            #    barcode_SS = barcode_SS[:int(cluster_elements[1]) - 1] + SS_ass + barcode[int(cluster_elements[2]):]
            SS_ass = "x" * len(cluster_elements[-1])
            barcode_SS = barcode_SS[:int(cluster_elements[1]) - 1] + SS_ass + barcode[int(cluster_elements[2]):]
    return (barcode, barcode_SS)


HCA_clusters_dico = {
    "110011":"H","11001":"H","100100011001":"H","1001100101":"H","110001":"h","1001000101":"H","100110010001":"H",
    "1000100110011":"H","10011":"H","10110101":"E","10001011":"h","1011001":"e","10110001":"e","10011011":"H",
    "10101":"E","100110011001":"H","101110011":"h","101011":"E","110010001001":"H","1101101":"e","110101":"E",
    "100011":"h","11001001":"H","10011001":"H","1100111":"e","1001110001":"e","101001":"e","10110011001":"H",
    "100100011":"H","100100010001":"H","101000110011":"H","100101":"e","10011111":"h","100011001001":"H",
    "100110011":"H","110010011001":"H","10001":"h","1001001":"h","11011101":"e","10010001001":"H","10111011":"E",
    "100010011":"H","100100011011":"H","1001101":"h","100011010001":"H","11011001":"H","10010001":"H","1000110011":"H",
    "110001001":"h","1100100101":"h","11011":"h","1000101101":"e","1111001":"E","1101000111":"m","111011001":"h",
    "1011001001":"h","111001":"e","10110011":"h","1111001101":"E","11110011":"E","1000100010011":"H","111011":"e",
    "110010011":"H","1100110011":"H","10010011":"H","10010101001":"e","10001001":"H","10011101":"e","1110010001":"h",
    "100111":"e","1101001":"E","1001000100101":"H","110011001":"H","10001111":"E","1100010011":"H","10101001":"E",
    "110110001":"H","1001001001":"h","10110010011":"H","1110010011":"h","10011011001":"H","11101101":"E","111001101":"e",
    "110001011":"h","111001011":"h","101111":"E","11001100101":"H","10111001":"e","100010010001":"H","1100101":"e",
    "10001101":"e","1000111101":"E","1001111":"E","1100011":"e","1010011":"e","100100111":"e","1010001":"e",
    "110111001":"H","110101001":"E","1000101":"e","10010001101":"h","100110001":"H","1001011":"E","1100100111":"h",
    "10010111":"e","110010001":"H","1100100011":"h","1110110001":"h","110110011":"H","100010011001":"H","10010101":"E",
    "11010001":"e","1010111":"E","1001100111":"h","1000100110001":"H","11011001001":"H","1011000101":"e","10110111":"e",
    "100010110101":"H","100010110001":"H","11100101001":"e","100010001":"H","10111":"E","1100110001":"H","1100101001":"e",
    "1100100010001":"H","11001011":"e","110111011":"H","1110011":"e","110111":"e","10110001001":"h","100100010011":"H",
    "1010011001":"H","10100011":"e","10001011001":"H","101101101":"e","101010010001":"m","1100011101":"e","10010001011":"h",
    "1001101101":"h","100010010101":"m","10001001001":"H","11101":"E","101100101":"h","10001000101":"e","1000111001":"m","110001101":"h","100111001":"e","111010011":"h","110100101":"e","10100101":"E","1010010001":"h","10011001001":"H","1000111011":"e","1101000101":"e","11000100101":"h","101010001":"e","110010010001":"H","101001101":"e","100101001":"e","110010010011":"H","100010101001":"e","10110010001":"H","11101001":"E","1110001":"E","10010010001":"H","1010001101":"e","1101100101":"h","1000111":"e","1010010101":"e","11000110001":"e","1010001001":"h","111001001":"e","10010010101":"e","1000110001":"h","111101":"E","1001101001":"h","11011011":"e","1101001011":"m","10001000111":"e","10010011001":"H","11001101":"h","1000110010001":"H","1000100011":"h","1001001111":"m","100011001":"H","101001011":"e","101010011":"h","1110001011":"e","11010111":"E","100010111":"e","101001111":"E","1100010001":"h","10010010011":"H","101001001":"h","111100010001":"E","11010011":"e","1101011":"E","1011001101":"H","101000101":"e","1100111001":"h","1010010011":"h","110110101":"e","100011011001":"H","101100110001":"H","1001010011":"h","1010101":"E","101100010001":"h","100010010011":"H","110100111":"e","110100010011":"h","1001110101":"e","10101101":"E","11000101":"e","101000101001":"e","1111000101":"e","10001001101":"h","11111":"E","100011011":"h","101001000101":"h","11100101":"E","1100101011":"m","1010100111":"E","1000100101":"h","101010001001":"h","101011101":"E","11010010001":"h","1110011011":"H","110011011":"H","1011011":"E","100010001101":"h","100010101":"e","100111101":"E","1100010101":"e","100011101":"E","101101":"E","11100011":"e","10010011011":"H","1110111":"e","1010010001001":"h","100010001001":"H","1001001000101":"h","1010110001":"e","1001001101":"h","11000111":"E","110010101":"e","11110101":"E","100110010011":"H","1101100011":"e","110001000101":"m","100111011":"h","1111101":"E","101100111":"h","1011101":"E","1101101001":"e","10101000111":"e","1000110111":"h","11001010001":"h","11110001":"E","1000101011":"e","11001100011":"H","1010100101":"E","1011100101":"E","11110111":"E","10001010011":"m","100100010101":"e","10001001011":"m","11011000101":"h","11000101001":"e","111111":"E","10010011101":"h","10100111":"e","11010101":"E","1101010001":"E","1100011001":"h","11001000101":"h","101011001":"e","1001011011":"e","1111011":"E","100110111":"H","100010110011":"H","110011101":"e","10010100011":"e","11111001":"E","111110011":"h","100110110011":"H","1101111":"E","1011111":"E","1110100011":"e","100110101":"e","1010100011":"e","10100010111":"e","111010101":"E","1010011011":"H","101001001001":"h","100100100011":"h","10011000101":"H","10101001001":"e","100011111":"e","10010001111":"m","10011110001":"E","1010010011001":"H","101011011":"e","1001010001":"e","1011010001":"e","11011010001":"h","10101111":"E","1001111001":"E","100010001011":"H","1001011101":"e","100100110001":"H","10100010001":"h","11001001001":"h","1100110101":"h","11001111":"e","1011010011":"e","110010001101":"h","10100101001":"e","10110110001":"h","1100100010011":"H","1011011001":"h","111010001":"E","1001001011":"m","1010101001":"e","1000100010001":"H","100110001001":"H","1001100011":"h","10011100101":"e","10100100111":"e","11000110011":"h","1011001011":"e","11110001001":"m","111000111":"E","1100101101":"m","10011001101":"H","10111000101":"e","100110010101":"H","11000100011":"h","111101011":"E","10100011001":"h","1110001001":"e","10101010011":"m","11100010001":"m","100101011":"e","11001011001":"h","11001000111":"m","1101010011":"h","11001001101":"H","1110011001":"h","1000110101":"E","1001011001":"h","110100011":"h","1000100111":"h","100100100101":"m","10011010001":"H","101110001":"E","111101001":"e","110011001001":"H","100100101":"e","10011100011":"e","11100111":"E","110010111":"h","10100110001":"H","10001010001":"h","110100010001":"h","10100100101":"e","11101011":"E","1011101001":"e","10011000111":"e","110100011001":"m","111000101":"E","1001100010001":"H","10101011":"E","100011000101":"e","1111001001":"e","1001100110001":"H","111100011":"e","101000111":"e","10100100011":"e","10001100011":"e","1110101001":"E","11010001101":"e","10101010001":"e","1110101":"E","100110001101":"e","10100010011":"h","101010111":"e","11101000101":"e","10100011011":"h","101010010011":"h","11010010011":"M","1100011011":"h","1010101101":"e","11001001011":"m","10100110011":"h","1101001001":"m","1101001101":"e","10010100101":"m","1110001101":"e","11100110001":"e","1010110011":"m","1110100101":"e","10101000101":"E","1000110011001":"H","1011010101":"E","101000100101":"m","11010001001":"h","100110110001":"H","10010110001":"e","101101011":"E","1001000111":"m","110010110001":"H","101101001":"E","111011011":"E","10001010101":"e","110001111":"E","10111101":"E","1010011101":"E","110110010001":"H","1010001111":"E","101110101":"E","10001100101":"h","1001000110001":"H","10001110001":"E","101001010001":"e","100101010001":"h","101100011":"e","101010101":"E","10100010101":"e","1001110011":"h","100101101":"E","10110100011":"E","1001010101":"e","100110001011":"h","1011110001":"e","10011001011":"h","111100101":"E","111110001":"E","110101011":"E","101000100011":"h","1001010111":"e","1010010111":"m","110101101":"E","10001101001":"m","1011000111":"E","1000101001":"e","10011010011":"h","11010011001":"m","1011100011":"e","110110111":"h","1010001011":"e","100101111":"m","1111010001":"E","1001101011":"e","1100010111":"e","1010110101":"e","101110010001":"h","11111011":"E","1010101011":"e","11010100101":"e","101000110001":"m","10010101101":"e","101111001":"E",
}


def calculate_the_sum_of_20(dico):
    values = np.array(list(dico.values()))
    return (np.sum(values / 20))


def calculate_the_sum_of_20_denom(dico, sum_norm_20):
    values = np.array(list(dico.values()))
    return (np.sqrt((np.sum(np.square(values - sum_norm_20)) / 20)))


def I(aa, aaj, aa_indexes):
    sum_norm_20 = calculate_the_sum_of_20(dico=aa_indexes[aaj])
    sum_of_20_denom = calculate_the_sum_of_20_denom(dico=aa_indexes[aaj], sum_norm_20=sum_norm_20)
    my_I = (aa_indexes[aaj][aa] - sum_norm_20) / sum_of_20_denom

    return (my_I)


def calculate_the_mean(seq, aa_indexes):
    dico = {}
    for aaj in aa_indexes:
        dico[aaj] = []
        for aa in seq:
#            if aa in dico : # MODIFIED BY PAUL
            dico[aaj].append(aa_indexes[aaj][aa])

    for j in dico:
        dico[j] = str(round(np.mean(dico[j]), 4))
    return (dico)