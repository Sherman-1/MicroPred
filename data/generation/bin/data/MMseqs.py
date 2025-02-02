import os
import subprocess
import sys
from pathlib import Path
import shutil
from Bio import SeqIO
import argparse


class MMseqs2:
    def __init__(self, threads, mmseqs2_path='mmseqs', cleanup=True):

        self.mmseqs2_path = mmseqs2_path
        self.threads = threads
        self.cleanup = cleanup
        self.dir = Path(os.getcwd()) / "mmseqs_tmp"
        self.dir.mkdir(parents=True, exist_ok=True)

        print(f"MMseqs running in {os.getcwd()}")

    def _run_command(self, command, log_file):
        print(f"Running command: {command}")
        with open(log_file, 'w') as log:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            log.write(stdout)
            log.write(stderr)
            
            if stderr:
                print(stderr, file=sys.stderr)
            
            if process.returncode != 0:
                raise RuntimeError(f"Command '{command}' failed with return code {process.returncode}")

    def createdb(self, fasta, log_file='createDB.log'):
        path = Path(fasta)
        if not path.exists():
            raise FileNotFoundError(f"FASTA file {fasta} does not exist.")
        command = f"{self.mmseqs2_path} createdb {path} {self.dir}/db"
        self._run_command(command, log_file)

    def cluster(self, coverage, identity, cov_mode, log_file='clustering.log'):
        command = (
            f"{self.mmseqs2_path} cluster {self.dir}/db {self.dir}/db_clu {self.dir}/tmp -c {coverage} --min-seq-id {identity} --cov-mode {cov_mode} --cluster-mode 2 --cluster-reassign -v 3 --threads {self.threads}"
        )
        self._run_command(command, log_file)

    def createtsv(self, tsv_file='result_seq_clu.tsv', log_file='createTsv.log'):
        command = f"{self.mmseqs2_path} createtsv {self.dir}/db {self.dir}/db {self.dir}/db_clu {self.dir}/{tsv_file}"
        self._run_command(command, log_file)
        if self.cleanup:
            self._cleanup_intermediate_files(tsv_file)

    def _cleanup_intermediate_files(self, tsv_file):
        files_to_keep = {self.dir / tsv_file}
        for file in self.dir.iterdir():
            if file not in files_to_keep:
                try:
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        shutil.rmtree(file)
                    print(f"{file} deleted")
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")

    def fasta2representativeseq(self, fasta_file : Path, cov : float, iden : float, cov_mode : int = 0) -> dict:
        
        self.createdb(fasta_file)
        self.cluster(coverage=cov, identity=iden, cov_mode=cov_mode)
        self.createtsv()

        with open(self.dir / "result_seq_clu.tsv") as tsv:
            representatives = {line.split()[0] for line in tsv.readlines()}
            
        full_fasta = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
        representatives_fasta = {k: full_fasta[k] for k in representatives}
        
        return representatives_fasta

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--fasta', help='Path to the input FASTA file')
    parser.add_argument('--writing_dir', default = ".", help='Directory to write the representative sequences')
    parser.add_argument('--cov', type=float, default = 0.5, help='Coverage threshold for clustering')
    parser.add_argument('--iden', type=float, default = 0.5, help='Identity threshold for clustering')
    parser.add_argument('--cov_mode', type=int, default=0, help='Coverage mode for clustering')
    parser.add_argument('--threads', type=int, default=os.cpu_count() - 2, help='Number of threads to use')

    args = parser.parse_args()

    mmseqs2_api = MMseqs2(threads=args.threads, cleanup=True)
    
    representatives = mmseqs2_api.fasta2representativeseq(args.fasta, args.cov, args.iden, args.cov_mode)

    SeqIO.write(representatives.values(), f"{args.writing_dir}/representative.fasta", "fasta")
    
    

   
