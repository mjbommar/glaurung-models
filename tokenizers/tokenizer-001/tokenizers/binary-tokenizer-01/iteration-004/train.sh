target/release/bbpe train \
		    --mode random \
		    --min-chunk-exp 2 \
		    --max-chunk-exp 5 \
		    --entropy-filter \
		    --progress \
		    --boundaries \
		    --max-token-length 16 \
		    --sample-rate 0.2 \
		    /home/ubuntu/src/glaurung-models/binary-sample/binaries/
