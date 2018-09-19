#echo "Event : "
python3 -W ignore sentence_kAlpha.py /home/ubuntu/flat/foliadocserver/$1 /home/ubuntu/flat/foliadocserver/$2 -Fo -s https://github.com/OsmanMutlu/rawtext/raw/master/prote10-Event.foliaset.xml
python3 -W ignore batch_kAlpha.py /home/ubuntu/flat/foliadocserver/$1 /home/ubuntu/flat/foliadocserver/$2 -Fo -s https://github.com/OsmanMutlu/rawtext/raw/master/prote10-Event.foliaset.xml
#echo "Participant : "
python3 -W ignore sentence_kAlpha.py /home/ubuntu/flat/foliadocserver/$1 /home/ubuntu/flat/foliadocserver/$2 -Fo -s https://github.com/OsmanMutlu/rawtext/raw/master/protes6-Participant.foliaset.xml
python3 -W ignore batch_kAlpha.py /home/ubuntu/flat/foliadocserver/$1 /home/ubuntu/flat/foliadocserver/$2 -Fo -s https://github.com/OsmanMutlu/rawtext/raw/master/protes6-Participant.foliaset.xml
#echo "Organizer : "
python3 -W ignore sentence_kAlpha.py /home/ubuntu/flat/foliadocserver/$1 /home/ubuntu/flat/foliadocserver/$2 -Fo -s https://github.com/OsmanMutlu/rawtext/raw/master/protes5-Organizer.foliaset.xml
python3 -W ignore batch_kAlpha.py /home/ubuntu/flat/foliadocserver/$1 /home/ubuntu/flat/foliadocserver/$2 -Fo -s https://github.com/OsmanMutlu/rawtext/raw/master/protes5-Organizer.foliaset.xml
#echo "Target : "
python3 -W ignore sentence_kAlpha.py /home/ubuntu/flat/foliadocserver/$1 /home/ubuntu/flat/foliadocserver/$2 -Fo -s https://github.com/OsmanMutlu/rawtext/raw/master/protes2-Target.foliaset.xml
python3 -W ignore batch_kAlpha.py /home/ubuntu/flat/foliadocserver/$1 /home/ubuntu/flat/foliadocserver/$2 -Fo -s https://github.com/OsmanMutlu/rawtext/raw/master/protes2-Target.foliaset.xml
