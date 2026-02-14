import subprocess

text = "Hello, this is a Festival female voice test."

scheme = f'(voice_cmu_us_slt_arctic_hts) (SayText "{text}")'

subprocess.run(
    ["festival"],
    input=scheme,
    text=True
)
