all: \
	generated/master.combined_json.json \
	generated/master.abi.json \
	generated/master.vy \
	generated/master.external_interface.vy

generated:
	mkdir -p generated

.PHONY: clean all

generated/%.combined_json.json: generated
	vyper --show-gas-estimates -f combined_json $*.vy > $@

generated/%.abi.json: generated
	vyper --show-gas-estimates -f abi $*.vy > $@

generated/%.vy: generated
	vyper --show-gas-estimates -f interface $*.vy > $@

generated/%.external_interface.vy: generated
	vyper --show-gas-estimates -f external_interface $*.vy > $@

clean:
	rm -rf generated
