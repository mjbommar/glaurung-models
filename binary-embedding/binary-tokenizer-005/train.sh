target/release/bbpe train \
		    --vocab-size 65536 \
		    --mode random \
		    --min-chunk-exp 2 \
		    --max-chunk-exp 6 \
		    --entropy-filter \
		    --progress \
		    --boundaries \
		    --pad-pow2 \
		    --max-token-length 16 \
		    --sample-rate 0.15 \
		    /home/ubuntu/src/glaurung-models/binary-sample/binaries/
