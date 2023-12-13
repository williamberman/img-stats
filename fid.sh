fidelity --gpu 0 --fid --input1 ~/out/muse_512/1.5 --input2 ~/val2017/  --samples-resize-and-crop 512 &> fid-1.5.txt & 
fidelity --gpu 0 --fid --input1 ~/out/muse_512/2.0 --input2 ~/val2017/  --samples-resize-and-crop 512 &> fid-2.0.txt &
fidelity --gpu 0 --fid --input1 ~/out/muse_512/3.0 --input2 ~/val2017/  --samples-resize-and-crop 512 &> fid-3.0.txt &
fidelity --gpu 0 --fid --input1 ~/out/muse_512/4.0 --input2 ~/val2017/  --samples-resize-and-crop 512 &> fid-4.0.txt &
fidelity --gpu 0 --fid --input1 ~/out/muse_512/5.0 --input2 ~/val2017/  --samples-resize-and-crop 512 &> fid-5.0.txt &