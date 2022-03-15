lang=$1
echo "1. clean the source sentences in the corpus"
python3 preprocess_scripts/clean_mustc.py --lang ${lang}
echo "2. convert raw data into tsv manifest"
python3 examples/speech_to_text/prep_mustc_data_raw.py --data-root data/mustc/ --tgt-lang ${lang}
echo "3. split audio files"
mkdir -p data/mustc/en-${lang}/segment/
python3 examples/speech_to_text/seg_mustc_data.py --data-root data/mustc/ --task st --lang ${lang} --output data/mustc/en-${lang}/segment/train --split train
python3 examples/speech_to_text/seg_mustc_data.py --data-root data/mustc/ --task st --lang ${lang} --output data/mustc/en-${lang}/segment/dev --split dev
python3 examples/speech_to_text/seg_mustc_data.py --data-root data/mustc/ --task st --lang ${lang} --output data/mustc/en-${lang}/segment/tst-COMMON --split tst-COMMON
python3 examples/speech_to_text/seg_mustc_data.py --data-root data/mustc/ --task st --lang ${lang} --output data/mustc/en-${lang}/segment/tst-HE --split tst-HE
echo "4. group by speaker"
python3 preprocess_scripts/group.py --lang ${lang}
echo "5. forced alignment"
cd data/mustc/en-${lang}/segment/
mfa align train english english train_align
mfa align dev english english dev_align
mfa align tst-COMMON english english tst-COMMON_align
mfa align tst-HE english english tst-HE_align
cd ../../../../
echo "6. convert textgrid format into tsv"
python3 preprocess_scripts/convert_format.py --lang ${lang}
echo "7. concatenate origin table and align table"
python3 preprocess_scripts/postprocess_raw.py --lang ${lang}
echo "8. learn dictionary"
python3 examples/speech_to_text/learn_dict_raw.py --data-root data/mustc/ --vocab-type unigram --vocab-size 10000 --tgt-lang ${lang}
echo "9. calculate the start and end indexs of aligned word sequence"
python3 preprocess_scripts/word_align_info_raw.py --lang ${lang}
echo "10. remove intermediate files"
rm data/mustc/en-${lang}/*_raw.tsv
rm data/mustc/en-${lang}/*_raw_seg.tsv
echo "Finish preprocess!"
