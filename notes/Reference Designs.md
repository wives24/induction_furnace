
[blog](https://www.seventransistorlabs.com/tmoranwms/Elec_IndHeat9.html)
- [youtube](https://www.youtube.com/watch?v=wKFnk4R54ZQ)
- [youtube](https://www.youtube.com/watch?v=Z385ysaaxL4) (custom)
- 1.5KW Induction Heater [youtube](https://www.youtube.com/watch?v=txw6YTetpK0)
	- careful of AC voltage rating at resonant frequency
	- KEMET R76 (342ARMS cts, 250V AC)
	- high performance [C500T](https://celem.com/product/c500t/)

Toroids
- https://www.digikey.com/en/products/detail/magnetics-a-division-of-spang-co/0077339A7/18626941
- TDK [PC95T96x20x70](https://product.tdk.com/en/search/ferrite/ferrite/ferrite-core/info?part_no=PC95T96x20x70)
- https://www.digikey.com/en/products/detail/epcos-tdk-electronics/B64290A0084X010/6161115
Ferrite
- Mag inc ZW48626TC https://www.mag-inc.com/Media/Magnetics/Datasheets/ZW48626TC.pdf (non stock, quote requested)
- ZP48613TC  [digikey](https://www.digikey.com/en/products/detail/magnetics-a-division-of-spang-co/ZP48613TC/18626685) (53mm ID in stock)
- ZP47313TC [digikey](https://www.digikey.com/en/products/detail/magnetics-a-division-of-spang-co/ZP47313TC/18626814) (37mm ID)


Film Resonant Caps
- [digikey](https://www.digikey.com/en/products/detail/kemet/R76PR3220SE30J/18306360)
	- [datasheet](https://content.kemet.com/datasheets/KEM_F3034_R76.pdf) has all deratings
- 


I think I want a soft saturating, high permeability core (with no gap), because I don't necessarily want any energy storage in the transformer, but rather just a impedance transform between the higher voltage, lower current primary
- KoolMu has a very slow soft saturation, which is great for CCM PFC or single quadrant operation as in a boost or flyback, but not great for resonant converters, because the nonlinearity would introduce hard to model harmonics


Current Sense Transformer
- ideally would work up to 100kHz of so
	- may be hard to ensure sufficient BW and extremely high turns ratio
	- core should not add significant inductance to the heating current loop
	- 1000A - 10mA (100k turns, I dont think so)

