/* ========================================================================
 * AppModel
 * Author: JoeHand
 * ========================================================================
 */

define([
    'backbone',
    'underscore'
], function (Backbone, _) {

    var App = Backbone.Model.extend({

        defaults: {
        },

        initialize: function(opt) {
            var user = opt.user;
        }
    });

    return App;
});
