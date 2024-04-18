init_file=$1
multi=$2
copies=$3
for ((i=0; i<=$copies; i++))
do
	        cp $init_file p1a_blobR_voxelS_9XYZ_$((($i*$multi)+9)).sub # adjust this file here
done
	
for file in *; do
    # Extract the number between "XYZ_" and ".sub"
    number=$(echo "$file" | grep -oP "XYZ_\K\d+(?=\.sub)")
    
    # Replace "BLOB_R=" with "BLOB_R=<extracted number>"
    sed -i "s/BLOB_R=/BLOB_R=$number/g" "$file" # if you are using something other than _9 as your base copying from, change the number here
done

