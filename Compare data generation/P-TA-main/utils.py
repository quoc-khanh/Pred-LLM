def format_row(row):
    return ", ".join([f"{col} is {row[col]}" for col in row.index])