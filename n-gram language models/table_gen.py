import pandas as pd
from tabulate import tabulate


# CSV dosyalarının isimleri ve Markdown dosyasının ismi
csv_files = ["sample_sentences.csv", "perplexity_results.csv"]
output_md_file = "tables.md"


# Başlıkları rapor dosyasına eklemek için metin
table_titles = {
    "sample_sentences.csv": "## Sample Sentences\n",
    "perplexity_results.csv": "## Perplexity Results\n"
}


# Markdown dosyasına yazma
with open(output_md_file, "a") as md_file:

    for csv_file in csv_files:
        # Dosyayı okuma
        try:
            df = pd.read_csv(csv_file)
            # Dosya başlığını ekleme
            md_file.write(f"\n{table_titles[csv_file]}\n")
            # Veriyi Markdown formatına dönüştürme ve dosyaya yazma
            md_file.write(tabulate(df, headers='keys', tablefmt='pipe'))
            md_file.write("\n\n--------------------------------------------------\n\n")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
