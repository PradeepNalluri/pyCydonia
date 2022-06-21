
for i in {2..106}; do 
    if [ $i -ge 10 ]; then
        filename=$(ls /research2/mtc/cp_traces/raw_traces/ | grep w${i}_)
        raw_path=/research2/mtc/cp_traces/raw_traces/$filename
        page_path=/research2/mtc/cp_traces/page_traces_4k/w${i}.csv
    else 
        filename=$(ls /research2/mtc/cp_traces/raw_traces/ | grep w0${i}_) 
        raw_path=/research2/mtc/cp_traces/raw_traces/$filename
        page_path=/research2/mtc/cp_traces/page_traces_4k/w0${i}.csv
    fi 
    python3 block_to_page.py $raw_path $page_path
done