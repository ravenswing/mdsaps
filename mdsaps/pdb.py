import pandas as pd


def process_atom(line):
    entry = {
        'record': line[0:6],
        'atom_id': line[6:11],
        'atom_name': line[12:16],
        'altloc': line[16],
        'res_name': line[17:21],
        'chain_id': line[21],
        'res_id': line[22:26],
        'icode': line[26],
        'x': line[30:38],
        'y': line[38:46],
        'z': line[46:54],
        'occupancy': line[54:60],
        'temp': line[60:66],
        'seg_id': line[66:76],
        'element': line[76:78],
        'charge': line[78:80],
    }

    entry = {k:[v] for k,v in entry.items()}

    return pd.DataFrame(entry)


def read_pdb(path, remove_seg_id=True):

    with open(path, "r") as f:
        lines = f.read().splitlines()

    df = pd.DataFrame(columns=[
        'record',
        'atom_id',
        'atom_name',
        'altloc',
        'res_name',
        'chain_id',
        'res_id',
        'icode',
        'x',
        'y',
        'z',
        'occupancy',
        'temp',
        'seg_id',
        'element',
        'charge',
    ])
    for line in lines:
        if line[0:4] == "ATOM":
            new_entry = process_atom(line)
            df = pd.concat([df, new_entry], ignore_index=True)
        elif line.split()[0] in ["TER", "END", "ENDMDL"]:
            df = pd.concat([df, pd.DataFrame([line.split()[0]], columns=["record"])], ignore_index=True)
        else:
            df = pd.concat([df, pd.DataFrame([line], columns=["record"])], ignore_index=True)

    df = df.astype({
        "atom_id": 'Int64',
        "res_id": 'Int64',
        "x": float,
        "y": float,
        "z": float,
        "occupancy": float,
        "temp": float,
    })

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    if remove_seg_id:
         df.loc[~df["seg_id"].isna(), "seg_id"] = ""

    return df


def format_name(name: str) -> str:
    # Full length names are left unchanged
    if len(name) == 4:
        return name
    # Otherwise align to second character
    else:
        return f" {name:<3}"


def format_atom(entry):

    new_line = (
        f"{entry.record:<6}"
        f"{entry.atom_id:>5.0f} "
        f"{format_name(entry.atom_name)}"
        f"{entry.altloc:1}"
        f"{entry.res_name:3} "
        f"{entry.chain_id:1}"
        f"{entry.res_id:>4.0f}"
        f"{entry.icode:1}   "
        f"{entry.x:>8.3f}"
        f"{entry.y:>8.3f}"
        f"{entry.z:>8.3f}"
        f"{entry.occupancy:>6.2f}"
        f"{entry.temp:>6.2f}"
        f"{entry.seg_id:>10}"
        f"{entry.element:>2}"
        f"{entry.charge:>2}\n"
    )

    return new_line


def write_pdb(df, out_path:str):
    output = []
    for _, line in df.iterrows():
        if line.record == "ATOM":
            output.append(format_atom(line))
        else:
            output.append(f"{line.record}\n")
    with open(out_path, 'w') as f:
        f.writelines(output)


def backbone_weights(df):
    # filter only backbone (removes TER, END)
    df = df[df['atom_name'].isin(['N', 'CA', 'C', 'O'])]
    # set occupancy to 1 ==> alignment weight
    df.loc[~df["occupancy"].isna(), "occupancy"] = 1
    # set temp factor (beta) to 1 ==> displacement calc. weight
    df.loc[~df["temp"].isna(), "temp"] = 1

    return df



