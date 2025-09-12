target/release/bbpe train \
		    --mode random \
		    --min-chunk-exp 2 \
		    --max-chunk-exp 8 \
		    --entropy-filter \
		    --progress \
		    --boundaries \
		    --max-token-length 32 \
		    --sample-rate 0.15 \
		    /home/ubuntu/src/glaurung-models/binary-sample/binaries/
