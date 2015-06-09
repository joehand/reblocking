/*! =======================================================================
 * Main JS
 * Author: JoeHand
 * ======================================================================== */

/*

proj4.defs("urn:ogc:def:crs:EPSG::26915", "+datum=WGS84 +no_defs +proj=utm +south +units=m +zone=36");



var geojson = [{'geometry': {'type': 'Polygon', 'coordinates': [[[305412.75999752525, 8022627.155497246], [305392.4976168405, 8022630.954693627], [305399.19024449866, 8022650.158977649], [305415.070022854, 8022647.590336719], [305412.75999752525, 8022627.155497246]]]}, 'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::3857'}}, 'type': 'Feature'}, {'geometry': {'type': 'Polygon', 'coordinates': [[[305411.72870420944, 8022609.620587965], [305388.3496718202, 8022612.795518289], [305389.7515385868, 8022623.074908677], [305392.4976168405, 8022630.954693627], [305412.75999752525, 8022627.155497246], [305412.6511477558, 8022626.192595449], [305412.09294581227, 8022611.679344853], [305411.72870420944, 8022609.620587965]]]}, 'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::3857'}}, 'type': 'Feature'},]
*/
var parcels = NAMESPACE.parcels,
    roads = NAMESPACE.roads,
    proj = NAMESPACE.proj;

console.log(proj);
console.log(parcels);
console.log(roads);

proj4.defs('urn:ogc:def:crs:EPSG::3857', proj);

var map = L.map('map').setView([-17.880209, 31.163120], 13);

L.tileLayer('http://{s}.tiles.mapbox.com/v3/joeahand.jc5epc4l/{z}/{x}/{y}.png', {
    attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="http://mapbox.com">Mapbox</a>',
    maxZoom: 18
}).addTo(map);

var parcelStyle = {
    "weight": 2,
    "opacity": 0.65
};

var roadStyle = function(feature) {
        console.log(feature);
        if (feature.properties.road == "true") {
            return {
                weight: 3,
                opacity: 0.7,
                "color":"#FF0000",
            };
        } else {
            return {
                weight: 2,
                "opacity": 0.65,
                "color":"#ff7800",
            };
        }
    };
if (parcels != null) {
    // Add geojson layer
    L.Proj.geoJson(parcels, {
        style: parcelStyle
    }).addTo(map);
    L.Proj.geoJson(roads, {
        style: roadStyle
    }).addTo(map);
    // Zoom to shapes
    map.fitBounds(L.Proj.geoJson(roads).getBounds());

}
