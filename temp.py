def plot():
    from matplotlib import pyplot as plt

    eval_loss = [
        0.028,
        0.027,
        0.028,
        0.029,
        0.031,
        0.032,
        0.032,
        0.033,
        0.034,
        0.035
    ]

    train_loss = [
        0.0323,
        0.0224,
        0.0195,
        0.016,
        0.0152,
        0.0159,
        0.0145,
        0.0153,
        0.0138,
        0.0134

    ]

    # Define the x-axis values
    epochs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    # Plot eval_loss in blue and label it as "Eval Loss"
    plt.figure(figsize=(10,7))
    plt.plot(epochs, eval_loss, label='Eval Loss', color='blue')  # Plot with defined epochs

    # Plot train_loss in red and label it as "Train Loss"
    plt.plot(epochs, train_loss, label='Train Loss', color='red')  # Plot with defined epochs

    # Add labels and a legend
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)  # Set x-axis ticks
    plt.ylim(0, 0.05)  # Set y-axis range
    plt.legend()

    # Show the plot
    plt.savefig('plot_equivariance.png')
    plt.show()

def csv2txt():
    """
    Convert CSV file to text (RAG knowledge base)
    """
    import csv

    # Define the path to your CSV file
    csv_file_path = '/home/donghee/Food/data/Data-Sweeney/molecules.csv'

    # Function to create natural language descriptions for attributes
    def create_description(row):
        attributes = [
            ('bond_stereo_count', 'includes {value} bond stereo centers'),
            ('undefined_atom_steroecenter_count', 'has {value} undefined atom stereo centers'),
            ('taste', 'is described as having a {value} taste'),
            ('functional_groups', 'contains functional groups such as {value}'),
            ('food_flavor_profile', 'has a food flavor profile of {value}'),
            ('supersweetdb_id', 'is registered in the SuperSweet database with ID {value}'),
            ('fema_number', 'has a FEMA number of {value}'),
            ('fooddb_id', 'is listed in the FoodDB with ID {value}'),
            ('common_name', '{value}'),
            ('hba_count', 'has {value} hydrogen bond acceptors'),
            ('synthetic', 'is {value}', lambda v: 'synthetic' if v == 'True' else 'not synthetic'),
            ('isotope_atom_count', 'has {value} isotope atoms'),
            ('bitters_id', 'is associated with Bitters ID {value}'),
            ('covalently_bonded_unit_count', 'consists of {value} covalently bonded units'),
            ('molecular_weight', 'has a molecular weight of {value}'),
            ('super_sweet', 'is {value}', lambda v: 'considered super sweet' if v == 'True' else 'not considered super sweet'),
            ('charge', 'has a charge of {value}'),
            ('flavornet_id', 'is known in FlavorNet with ID {value}'),
            ('fenoroli_and_os', 'is featured in Fenaroli\'s Handbook with {value}'),
            ('exact_mass', 'has an exact mass of {value}'),
            ('pubchem_id', 'has a PubChem ID of {value}'),
            ('bitter', 'is {value}', lambda v: 'bitter' if v == 'True' else 'not bitter'),
            ('iupac_name', 'is also known by its IUPAC name, {value}'),
            ('volume3d', 'has a 3D volume of {value} cubic units'),
            ('unknown_natural', 'is {value}', lambda v: 'possibly natural' if v == 'True' else 'of unknown natural origin'),
            ('odor', 'has an odor described as {value}'),
            ('num_rotatablebonds', 'has {value} rotatable bonds'),
            ('smile', 'can be represented by the SMILES notation {value}'),
            ('inch', 'has an InChI string of {value}'),
            ('undefined_bond_stereocenter_count', 'has {value} undefined bond stereo centers'),
            ('defined_bond_stereocenter_count', 'has {value} defined bond stereo centers'),
            ('xlogp', 'has an XLogP of {value}'),
            ('topological_polor_surfacearea', 'has a topological polar surface area of {value}'),
            ('cas_id', 'is registered under the CAS ID {value}'),
            ('natural', 'is {value}', lambda v: 'naturally occurring' if v == 'True' else 'not naturally occurring'),
            ('flavor_profile', 'offers a flavor profile described as {value}'),
            ('hbd_count', 'has {value} hydrogen bond donors'),
            ('fema_falvor_profile', 'has a FEMA flavor profile of {value}'),
            ('complexity', 'has a complexity rating of {value}'),
            ('heavy_atom_count', 'is made up of {value} heavy atoms'),
            ('defined_atom_sterocenter_count', 'has {value} defined atom stereo centers'),
            ('monoisotopic_mass', 'has a monoisotopic mass of {value}'),
            ('atom_stereo_count', 'includes {value} atom stereo centers'),
        ]
        
        # description = f"Molecule {row.get('id', 'with no ID')} - "
        # if row.get('common_name'):
        #     description += f"known as {row['common_name']}, "
        description = f"Molecule {row['common_name']}"
        
        descriptions = [description]
        
        # for key, template, formatter=lambda v: v in attributes:
        #     if row.get(key, '').strip():
        #         value = formatter(row[key].strip())
        #         descriptions.append(template.format(value=value))
        
        # return ' '.join(descriptions)
        for key, template, *optional in attributes:
            if row.get(key, '').strip():
                value = row[key].strip()
                if optional:
                    formatter = optional[0]
                    value = formatter(value)
                description += template.format(value=value) + " "
    
        return description.strip()


    # Open the CSV file for reading
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(csv_file)
        
        # Iterate over each row in the CSV
        for row in csv_reader:
            # Generate the description
            text_description = create_description(row)
            
            # Define the output file name using the molecule's ID
            output_file_name = f"/home/donghee/Food/data/RAG_documents/molecule_{row['id']}.txt"
            
            # Save the descriptive text to a text file
            with open(output_file_name, mode='w', encoding='utf-8') as text_file:
                text_file.write(text_description)

def db2txt():
    """
    Convert from Database to txt
    """
    import sqlite3

    # Define the path to your database file
    db_file_path = '/home/donghee/Food/data/Data-Sweeney/flavordb.db'

    # Define the function to create natural language descriptions for database rows
    def create_description(row, columns):
        attributes = [
            ('bond_stereo_count', ', includes {value} bond stereo centers'),
            ('undefined_atom_steroecenter_count', ', has {value} undefined atom stereo centers'),
            ('taste', ', is described as having a {value} taste'),
            ('functional_groups', ', contains functional groups such as {value}'),
            ('food_flavor_profile', ', has a food flavor profile of {value}'),
            ('supersweetdb_id', ', is registered in the SuperSweet database with ID {value}'),
            ('fema_number', ', has a FEMA number of {value}'),
            ('fooddb_id', ', is listed in the FoodDB with ID {value}'),
            # ('common_name', '{value}'),
            ('hba_count', ', has {value} hydrogen bond acceptors'),
            ('synthetic', ', is {value}', lambda v: 'synthetic' if bool(v) == True else 'not synthetic'),
            ('isotope_atom_count', ', has {value} isotope atoms'),
            ('bitters_id', ', is associated with Bitters ID {value}'),
            ('covalently_bonded_unit_count', ', consists of {value} covalently bonded units'),
            ('molecular_weight', ', has a molecular weight of {value}'),
            ('super_sweet', 'is {value}', lambda v: ', considered super sweet' if bool(v) == True else 'not considered super sweet'),
            ('charge', ', has a charge of {value}'),
            ('flavornet_id', ', is known in FlavorNet with ID {value}'),
            ('fenoroli_and_os', ', is featured in Fenaroli\'s Handbook with {value}'),
            ('exact_mass', ', has an exact mass of {value}'),
            ('pubchem_id', ', has a PubChem ID of {value}'),
            ('bitter', ', is {value}', lambda v: 'bitter' if bool(v) == True else 'not bitter'),
            ('iupac_name', ', is also known by its IUPAC name, {value}'),
            ('volume3d', ', has a 3D volume of {value} cubic units'),
            ('unknown_natural', ', is {value}', lambda v: 'possibly natural' if bool(v) == True else 'of unknown natural origin'),
            ('odor', ', has an odor described as {value}'),
            ('num_rotatablebonds', ', has {value} rotatable bonds'),
            ('smile', ', can be represented by the SMILES notation {value}'),
            ('inch', ', has an InChI string of {value}'),
            ('undefined_bond_stereocenter_count', ', has {value} undefined bond stereo centers'),
            ('defined_bond_stereocenter_count', ', has {value} defined bond stereo centers'),
            ('xlogp', ', has an XLogP of {value}'),
            ('topological_polor_surfacearea', ', has a topological polar surface area of {value}'),
            ('cas_id', ', is registered under the CAS ID {value}'),
            ('natural', ', is {value}', lambda v: 'naturally occurring' if v == 'True' else 'not naturally occurring'),
            ('flavor_profile', ', offers a flavor profile described as {value}'),
            ('hbd_count', ', has {value} hydrogen bond donors'),
            ('fema_falvor_profile', ', has a FEMA flavor profile of {value}'),
            ('complexity', ', has a complexity rating of {value}'),
            ('heavy_atom_count', ', is made up of {value} heavy atoms'),
            ('defined_atom_sterocenter_count', ', has {value} defined atom stereo centers'),
            ('monoisotopic_mass', ', has a monoisotopic mass of {value}'),
            ('atom_stereo_count', ', includes {value} atom stereo centers'),
        ]
        
        description = f"Molecule "
        if 'common_name' in columns:
            common_name_index = columns.index('common_name')
            description += f"{row[common_name_index]}"

        for attribute in attributes:
            key, template, *optional = attribute
            if key in columns:
                value_index = columns.index(key)
                value = row[value_index]
                if value:
                    if optional:
                        formatter = optional[0]
                        value = formatter(value)
                    description += template.format(value=value) + " "

        return description.strip()

    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Fetch all rows from the 'molecules' table
    cursor.execute("SELECT * FROM molecules")
    rows = cursor.fetchall()

    # Fetch the column names
    cursor.execute("PRAGMA table_info(molecules)")
    columns = [desc[1] for desc in cursor.fetchall()]

    # Iterate over each row and create a text file with the natural language description
    for index, row in enumerate(rows):
        text_description = create_description(row, columns)
        
        # Define the output file name using an identifier from the row or index
        output_file_name = f"/home/donghee/Food/data/RAG_documents/molecule_id_{row[0]}.txt"  # Assuming the first column is a unique identifier
        
        # Save the descriptive text to a text file
        with open(output_file_name, mode='w', encoding='utf-8') as text_file:
            text_file.write(text_description)

    # Close the database connection
    conn.close()

def main():
    import sqlite3

    # Path to your SQLite database file
    db_file_path = '/home/donghee/Food/data/FlavorDB-Scrape/flavordb.db'

    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)

    # Create a cursor object
    cursor = conn.cursor()

    # Execute a query to fetch molecule IDs and their common names
    cursor.execute("SELECT id, common_name FROM molecules")

    # Fetch all rows from the cursor into a list
    id_common_name_pairs = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Convert the list of pairs into a dictionary
    id_to_common_name = {str(id): common_name for id, common_name in id_common_name_pairs}

    import csv
    import os

    # Path to the directory containing your text files
    text_files_directory = '/home/donghee/Food/data/RAG_documents'

    # The path for the output CSV file
    output_csv_path = './data/RAG_molecules_combined.csv'

    # Open the output CSV file for writing
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        # Write the header
        writer.writerow(['title', 'text'])
        
        # Iterate through each text file in the directory
        for filename in os.listdir(text_files_directory):
            if filename.endswith(".txt"):
                # Extract the ID from the filename (assuming the format 'molecule_id_[id].txt')
                molecule_id = filename.split('_')[-1].split('.')[0]
                
                # Lookup the common name using the extracted ID
                common_name = id_to_common_name[molecule_id]
                
                # Read the content of the text file
                file_path = os.path.join(text_files_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Write a row to the CSV file with the common name and content
                writer.writerow([common_name, content])



if __name__ == '__main__':
    main()
